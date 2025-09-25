# filename: get_args.py

import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for Pixel-Space Adversarial Purifier.")

    # --- Core Model and Dataset Arguments ---
    parser.add_argument("--dataset", type=str, default="cifar10", help="The dataset to use (cifar10).")
    parser.add_argument(
        "--pretrained_purifier_model", type=str, default="google/ddpm-cifar10-32",
        help="Base DDPM model for the purifier, from huggingface.co/models."
    )
    parser.add_argument("--surrogate_model", type=str, default="resnet56_cifar10",
                        help="The surrogate classifier model to generate adversarial examples.")

    # --- Output and Logging ---
    parser.add_argument("--output_dir", type=str, default="logs/OSCP-CIFAR10-PIXEL",
                        help="The output directory for checkpoints and logs.")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--tracker_project_name", type=str, default="OCSP-Pixel-LoRA")

    # --- Checkpointing ---
    parser.add_argument("--checkpointing_steps", type=int, default=1000)
    parser.add_argument("--checkpoints_total_limit", type=int, default=5)
    parser.add_argument("--resume_from_checkpoint", type=str, default="latest")

    # --- Training Parameters ---
    parser.add_argument("--seed", type=int, default=3407, help="A seed for reproducible training.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=5000,
                        help="Total number of training steps. Overrides num_train_epochs.")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size (per device).")
    parser.add_argument("--dataloader_num_workers", type=int, default=8)

    # --- Optimizer and Scheduler ---
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="constant_with_warmup")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--adam_weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    # --- GAND Adversarial Training Parameters ---
    parser.add_argument("--eps", type=float, default=8/255, help="PGD attack epsilon (pixel space).")
    parser.add_argument("--attack_iter", type=int, default=10, help="Number of PGD attack iterations.")
    parser.add_argument("--eps_iter", type=float, default=2/255, help="PGD attack step size.")
    parser.add_argument("--lmd", type=float, default=0.001, help="Weight for the CIG loss term.")
    parser.add_argument("--N", type=int, default=2, help="Timestep skip factor for GAND training.")

    # --- LoRA Parameters ---
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)

    # --- Diffusion Model Parameters ---
    parser.add_argument("--num_ddim_timesteps", type=int, default=50)
    parser.add_argument("--loss_type", type=str, default="l2", choices=["l2", "huber"])

    # --- Performance and Precision ---
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing.")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Enable xformers.")

    args = parser.parse_args()
    return args