# Pixel-Space Adversarial Purifier for CIFAR-10

This project adapts the "One-Step Control Purification (OSCP)" methodology to work effectively on the CIFAR-10 dataset.

The core problem with the original approach was using a high-resolution Stable Diffusion model, which treated upsampled CIFAR-10 images as "out-of-distribution" and destructively "corrected" them.

This new implementation resolves that issue by replacing the Stable Diffusion backbone with a `UNet2DModel` pre-trained directly on CIFAR-10. The entire purification process now happens in the native 32x32 pixel space, eliminating the need for a VAE, upsampling, or downsampling.

The model used is - https://huggingface.co/google/ddpm-cifar10-32

## Project Structure

- `train_lora.py`: The main script for training the purifier using the GAND objective.
- `evaluate.py`: A unified script for evaluating the trained purifier's performance on clean and adversarial images.
- `dataset.py`: Handles loading the CIFAR-10 dataset.
- `archs.py`: Loads the pre-trained CIFAR-10 victim classifier (ResNet-56).
- `get_args.py`: Manages command-line arguments for training.
- `train.sh`: Example script to launch the training process.
- `evaluate.sh`: Example script to run the evaluation.
- `requirements.txt`: Required Python packages.

## How to Run

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Train the Purifier:**
    Modify `train.sh` if needed, then run:
    ```bash
    bash train.sh
    ```
    Checkpoints will be saved to the directory specified in the script (e.g., `./logs/OSCP-CIFAR10-PIXEL`).

3.  **Evaluate the Purifier:**
    Update the `--lora_input_dir` path in `evaluate.sh` to point to your latest checkpoint, then run:
    ```bash
    bash evaluate.sh
    ```
    Results will be saved to the specified output directory.