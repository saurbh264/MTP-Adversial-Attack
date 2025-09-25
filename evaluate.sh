#!/bin/bash

# Script to evaluate a trained purifier checkpoint on the FULL CIFAR-10 TEST SET.

# Path to your trained model checkpoint
export CHECKPOINT_PATH="./logs/OSCP-CIFAR10-PIXEL/checkpoint-5000"
export DATASET_NAME="cifar10"
export CLASSIFIER_NAME="resnet56_cifar10"
export OUTPUT_DIR="evaluation_results_sweep_FULL" # New output directory for full results

# Define the range of strengths you want to test
STRENGTHS=(0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4)

echo "Starting evaluation sweep for checkpoint: $CHECKPOINT_PATH"
echo "Testing on the FULL 10,000 image CIFAR-10 test set."
echo "Testing strengths: ${STRENGTHS[@]}"
echo "--------------------------------------------------"

# Loop through each strength and run the evaluation
for STRENGTH in "${STRENGTHS[@]}"
do
  echo "RUNNING EVALUATION FOR STRENGTH: $STRENGTH"
  
  python3 evaluate.py \
      --lora_input_dir=$CHECKPOINT_PATH \
      --output_dir=$OUTPUT_DIR \
      --dataset=$DATASET_NAME \
      --classifier=$CLASSIFIER_NAME \
      --num_samples=10000 \
      --batch_size=50 \
      --strength=$STRENGTH \
      --device="cuda:0"

  echo "--------------------------------------------------"
done

echo "Evaluation sweep complete. Results are in the '$OUTPUT_DIR' directory."