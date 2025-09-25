# filename: train_lora.py

import os
import math
import shutil
import logging
import diffusers
import accelerate
import transformers
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from pathlib import Path
from tqdm.auto import tqdm

from get_args import parse_args
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from diffusers import DDPMScheduler, UNet2DModel
from diffusers.optimization import get_scheduler
from advertorch.attacks import LinfPGDAttack

from dataset import get_dataset, get_num_classes
from archs import get_archs
from ddim_solver import DDIMSolver
from utils import append_dims, scalings_for_boundary_conditions, get_predicted_original_sample

logger = get_logger(__name__)

def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=str(logging_dir))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Scheduler and Solver
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_purifier_model)
    solver = DDIMSolver(
        noise_scheduler.alphas_cumprod.numpy(),
        timesteps=noise_scheduler.config.num_train_timesteps,
        ddim_timesteps=args.num_ddim_timesteps,
    )
    # --- FIX: Move the solver's internal tensors to the correct device ---
    solver.to(accelerator.device)

    # 2. Load U-Net Models (Teacher and Student)
    teacher_unet = UNet2DModel.from_pretrained(args.pretrained_purifier_model)
    unet = UNet2DModel.from_pretrained(args.pretrained_purifier_model)
    unet.train()
    teacher_unet.requires_grad_(False)

    # 3. Add LoRA to the student U-Net
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    unet = get_peft_model(unet, lora_config)

    # 4. Handle Mixed Precision and Device Placement
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    teacher_unet.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # 5. Optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(), lr=args.learning_rate, weight_decay=args.adam_weight_decay
    )

    # 6. Dataset and DataLoader
    dataset = get_dataset(args.dataset, split="train")
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
    )

    # 7. LR Scheduler
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # 8. Prepare for training with Accelerate
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # 9. Prepare Classifier and Adversary
    classifier = get_archs(args.surrogate_model, args.dataset).to(accelerator.device).eval()
    adversary = LinfPGDAttack(
        classifier,
        loss_fn=torch.nn.CrossEntropyLoss(),
        eps=args.eps,
        nb_iter=args.attack_iter,
        eps_iter=args.eps_iter,
        rand_init=True,
        clip_min=-1.0, # Images are in [-1, 1] range
        clip_max=1.0,
        targeted=False,
    )

    # 10. Training Loop
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    global_step = 0
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(args.num_train_epochs):
        for step, (image, label) in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Generate adversarial examples in pixel space
                adv_images = adversary(image, label)

                # Sample timesteps
                bsz = image.shape[0]
                topk = noise_scheduler.config.num_train_timesteps // args.num_ddim_timesteps
                index = torch.randint(0, args.num_ddim_timesteps // args.N, (bsz,), device=accelerator.device).long()
                start_timesteps = solver.ddim_timesteps[index]
                timesteps = start_timesteps - topk
                timesteps = torch.where(timesteps < 0, 0, timesteps).long()

                # Get boundary scaling
                c_skip_start, c_out_start = [append_dims(x, image.ndim) for x in scalings_for_boundary_conditions(start_timesteps)]
                c_skip, c_out = [append_dims(x, image.ndim) for x in scalings_for_boundary_conditions(timesteps)]

                # Add noise to adversarial images
                noise = torch.randn_like(adv_images)
                noisy_model_input = noise_scheduler.add_noise(adv_images, noise, start_timesteps)

                # Get student model prediction
                model_pred_noise = unet(noisy_model_input, start_timesteps).sample
                pred_x_0 = get_predicted_original_sample(model_pred_noise, start_timesteps, noisy_model_input, "epsilon", noise_scheduler.alphas_cumprod)
                model_pred = c_skip_start * noisy_model_input + c_out_start * pred_x_0

                # Get teacher model prediction
                with torch.no_grad():
                    teacher_pred_noise = teacher_unet(noisy_model_input.to(weight_dtype), start_timesteps).sample
                    pred_x0_teacher = get_predicted_original_sample(teacher_pred_noise, start_timesteps, noisy_model_input, "epsilon", noise_scheduler.alphas_cumprod)
                    x_prev = solver.ddim_step(pred_x0_teacher, teacher_pred_noise, index)
                    
                    # Get target for distillation
                    target_pred_noise = teacher_unet(x_prev.to(weight_dtype), timesteps).sample
                    pred_x0_target = get_predicted_original_sample(target_pred_noise, timesteps, x_prev, "epsilon", noise_scheduler.alphas_cumprod)
                    target = c_skip * x_prev + c_out * pred_x0_target

                # Calculate loss
                # GAND Loss: Match student output to teacher's DDIM step output
                loss_gand = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                # CIG Loss: Pull student output towards original clean image
                loss_cig = F.mse_loss(model_pred.float(), image.float(), reduction="mean")
                loss = loss_gand + args.lmd * loss_cig

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.unwrap_model(unet).save_pretrained(save_path)
                        logger.info(f"Saved checkpoint to {save_path}")

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    # Save final model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        unet.save_pretrained(args.output_dir)
    
    accelerator.end_training()

if __name__ == "__main__":
    arguments = parse_args()
    main(arguments)