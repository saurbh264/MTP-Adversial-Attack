#!/bin/bash

# Script to train the pixel-space purifier for CIFAR-10

export MODEL_NAME="google/ddpm-cifar10-32"
export OUTPUT_DIR="./logs/OSCP-CIFAR10-PIXEL"
export DATASET_NAME="cifar10"
export SURROGATE_CLASSIFIER="resnet56_cifar10"

accelerate launch train_lora.py \
    --pretrained_purifier_model=$MODEL_NAME \
    --output_dir=$OUTPUT_DIR \
    --dataset=$DATASET_NAME \
    --surrogate_model=$SURROGATE_CLASSIFIER \
    --lora_rank=32 \
    --learning_rate=1e-4 \
    --max_train_steps=5000 \
    --train_batch_size=32 \
    --gradient_accumulation_steps=1 \
    --checkpointing_steps=1000 \
    --checkpoints_total_limit=5 \
    --lr_scheduler="constant_with_warmup" \
    --lr_warmup_steps=100 \
    --seed=42 \
    --report_to="tensorboard" \
    --resume_from_checkpoint="latest" \
    --mixed_precision="no"