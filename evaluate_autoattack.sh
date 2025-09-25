#!/bin/bash

# Script to evaluate a trained purifier against the AutoAttack benchmark.

# Path to your trained model checkpoint
export CHECKPOINT_PATH="./logs/OSCP-CIFAR10-PIXEL/checkpoint-5000"
export DATASET_NAME="cifar10"
export CLASSIFIER_NAME="resnet56_cifar10"
export OUTPUT_DIR="evaluation_results_autoattack"

# Set the single best strength found from the PGD sweep
export STRENGTH=0.05

# Set number of samples. Start with a smaller number as AutoAttack is slow.
# Increase to 1000 or more for the final paper-ready result.
export NUM_SAMPLES=10000

echo "Starting AutoAttack evaluation for checkpoint: $CHECKPOINT_PATH"
echo "Testing with optimal strength: $STRENGTH on $NUM_SAMPLES samples."
echo "--------------------------------------------------"

python3 evaluate.py \
    --lora_input_dir=$CHECKPOINT_PATH \
    --output_dir=$OUTPUT_DIR \
    --dataset=$DATASET_NAME \
    --classifier=$CLASSIFIER_NAME \
    --attack_method "autoattack" \
    --num_samples=$NUM_SAMPLES \
    --batch_size=50 \
    --strength=$STRENGTH \
    --device="cuda:0"

echo "--------------------------------------------------"
echo "AutoAttack evaluation complete. Results are in the '$OUTPUT_DIR' directory."