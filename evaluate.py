# filename: evaluate.py

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import random
import os
import argparse
from tqdm.auto import tqdm
from PIL import Image
from torch.utils.data import DataLoader

from diffusers import DDPMScheduler, UNet2DModel
from advertorch.attacks import LinfPGDAttack
from peft import PeftModel
from autoattack import AutoAttack

from archs import get_archs
from dataset import get_dataset, get_normalize_layer
from utils import si, get_predicted_original_sample, scalings_for_boundary_conditions, append_dims

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation script for Pixel-Space Purifier.")
    parser.add_argument("--attack_method", default="pgd", type=str, choices=["pgd", "autoattack"])
    parser.add_argument("--dataset", default="cifar10", type=str)
    parser.add_argument("--lora_input_dir", required=True, type=str)
    parser.add_argument("--output_dir", default="output_cifar10/", type=str)
    parser.add_argument("--num_samples", default=1000, type=int)
    parser.add_argument("--batch_size", default=50, type=int)
    parser.add_argument("--strength", default=0.25, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--classifier", default="resnet56_cifar10", type=str)
    parser.add_argument("--eps", type=float, default=8/255)
    parser.add_argument("--attack_iter", type=int, default=20)
    parser.add_argument("--attack_step_size", type=float, default=2/255)
    parser.add_argument("--device", default="cuda:0", type=str)
    return parser.parse_args()

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    args = parse_args()
    seed_everything(args.seed)

    save_path = os.path.join(args.output_dir, os.path.basename(os.path.normpath(args.lora_input_dir)), f"attack_{args.attack_method}_strength_{args.strength}")
    os.makedirs(save_path, exist_ok=True)
    
    device = torch.device(args.device)
    classifier = get_archs(args.classifier, args.dataset).to(device).eval()
    
    normalize_layer = get_normalize_layer(args.dataset)
    attack_classifier = nn.Sequential(normalize_layer, classifier).to(device).eval()

    # --- FIX: Corrected the typo in the model name ---
    base_model_path = "google/ddpm-cifar10-32"
    base_purifier = UNet2DModel.from_pretrained(base_model_path)
    purifier = PeftModel.from_pretrained(base_purifier, args.lora_input_dir).to(device)
    purifier.eval()

    scheduler = DDPMScheduler.from_pretrained(base_model_path)

    # The 'test' split from our dataset loader provides normalized images for the classifier
    # but attacks need the [0, 1] range.
    dataset_for_attack = get_dataset(args.dataset, split='train') # Using train transform for [0,1] range
    dataset = get_dataset(args.dataset, split='test')
    
    indices = torch.randperm(len(dataset))[:args.num_samples].tolist()
    subset_dataset = torch.utils.data.Subset(dataset, indices)
    test_loader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=False)
    
    if args.attack_method == "pgd":
        adversary = LinfPGDAttack(
            attack_classifier, loss_fn=torch.nn.CrossEntropyLoss(),
            eps=args.eps, nb_iter=args.attack_iter, eps_iter=args.attack_step_size,
            rand_init=True, targeted=False, clip_min=0.0, clip_max=1.0
        )
    elif args.attack_method == "autoattack":
        adversary = AutoAttack(attack_classifier, norm='Linf', eps=args.eps, version='standard', device=device)
    
    clean_acc, robust_acc, total_attacked = 0, 0, 0
    
    for i, (x_normalized, y) in enumerate(tqdm(test_loader, desc=f"Evaluating with {args.attack_method.upper()}")):
        x_normalized, y = x_normalized.to(device), y.to(device)
        
        mu_c = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(device)
        sigma_c = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1).to(device)
        x_0_1 = x_normalized * sigma_c + mu_c
        
        clean_preds = classifier(x_normalized).argmax(1)
        clean_acc += (clean_preds == y).sum().item()
        
        # Only attack images that are correctly classified
        correct_mask = (clean_preds == y)
        x_to_attack = x_0_1[correct_mask]
        y_to_attack = y[correct_mask]
        total_attacked += x_to_attack.shape[0]

        if x_to_attack.shape[0] > 0:
            if args.attack_method == "pgd":
                x_adv_0_1 = adversary.perturb(x_to_attack, y_to_attack)
            elif args.attack_method == "autoattack":
                x_adv_0_1 = adversary.run_standard_evaluation(x_to_attack, y_to_attack)
        
            x_adv_normalized = (x_adv_0_1 - mu_c) / sigma_c
        
            with torch.no_grad():
                x_adv_purifier_input = (x_adv_0_1 - 0.5) / 0.5
                t_start = int(args.strength * (scheduler.config.num_train_timesteps - 1))
                timesteps = torch.tensor([t_start] * x_adv_purifier_input.shape[0], device=device)
                
                noise = torch.randn_like(x_adv_purifier_input)
                noisy_x = scheduler.add_noise(x_adv_purifier_input, noise, timesteps)
                
                pred_noise = purifier(noisy_x, timesteps).sample
                pred_x0 = get_predicted_original_sample(pred_noise, timesteps, noisy_x, "epsilon", scheduler.alphas_cumprod)
                c_skip, c_out = [append_dims(x, noisy_x.ndim) for x in scalings_for_boundary_conditions(timesteps)]
                purified_x_purifier_output = c_skip * noisy_x + c_out * pred_x0
                
                purified_x_0_1 = (purified_x_purifier_output.clamp(-1, 1) * 0.5) + 0.5
                robust_x = (purified_x_0_1 - mu_c) / sigma_c

            robust_preds = classifier(robust_x).argmax(1)
            robust_acc += (robust_preds == y_to_attack).sum().item()
        
            if i == 0:
                for j in range(min(5, x_normalized.shape[0])):
                    si(x_normalized[j], os.path.join(save_path, f'clean_{j}.png'))
                    si(x_adv_normalized[j], os.path.join(save_path, f'adv_{j}.png'))
                    si(robust_x[j], os.path.join(save_path, f'robust_{j}.png'))

    total = len(subset_dataset)
    # Attack success rate is how many of the correctly classified images were fooled
    attack_success_rate = 1 - (robust_acc / total_attacked) if total_attacked > 0 else 0
    
    stats = {
        "strength": args.strength,
        "attack": args.attack_method,
        "clean_accuracy": clean_acc / total,
        "attack_success_rate": attack_success_rate,
        "robust_accuracy": robust_acc / total_attacked if total_attacked > 0 else 0,
    }
    pd.DataFrame(stats, index=[0]).to_csv(os.path.join(save_path, 'stats.csv'))
    print(f"\n--- Evaluation Complete for Strength {args.strength} ---")
    print(pd.DataFrame(stats, index=[0]))

if __name__ == '__main__':
    main()