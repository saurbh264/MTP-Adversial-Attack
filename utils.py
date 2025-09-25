# filename: utils.py

import torch
import numpy as np
from PIL import Image

# --- Helper functions from Latent Consistency Models ---

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less.")
    return x[(...,) + (None,) * dims_to_append]

def scalings_for_boundary_conditions(timestep, timestep_scaling=10.0):
    """
    Calculates the scalings for the boundary conditions of the ODE solver.
    """
    c_skip = timestep_scaling / (timestep**2 + timestep_scaling**2)
    c_out = timestep / (timestep**2 + timestep_scaling**2) ** 0.5
    return c_skip, c_out

def get_predicted_original_sample(
    model_output, timesteps, sample, prediction_type, alphas_cumprod
):
    """
    Get the predicted original sample from the model output (noise prediction).
    """
    alphas_cumprod = alphas_cumprod.to(sample.device)
    alpha_t = alphas_cumprod[timesteps.long()]
    
    while alpha_t.ndim < sample.ndim:
        alpha_t = alpha_t.unsqueeze(-1)
        
    beta_t = 1 - alpha_t
    
    if prediction_type == "epsilon":
        pred_original_sample = (sample - beta_t.sqrt() * model_output) / alpha_t.sqrt()
    else:
        raise ValueError(f"Unsupported prediction_type: {prediction_type}")
        
    return pred_original_sample

# --- Image Saving Utility ---

def si(img, path, is_BCHW=True):
    """
    Saves a torch tensor image.
    Handles denormalization from [-1, 1] or a classifier-specific range.
    """
    if is_BCHW:
        img = img.permute(1, 2, 0) # Convert from CHW to HWC

    img_np = img.detach().cpu().numpy()

    # Heuristic to check if image is in [-1, 1] or [0, 1] range based on min value
    if img_np.min() < -0.1:
        # Denormalize from [-1, 1] to [0, 255]
        img_denorm = ((img_np + 1) / 2 * 255).astype(np.uint8)
    else:
        # Denormalize from a custom range (like classifier's) to [0, 255]
        # This requires finding the true min/max and scaling
        min_val, max_val = img_np.min(), img_np.max()
        if max_val > min_val:
            img_scaled = (img_np - min_val) / (max_val - min_val)
            img_denorm = (img_scaled * 255).astype(np.uint8)
        else: # Handle zero-range images
            img_denorm = (img_np * 255).astype(np.uint8)

    Image.fromarray(img_denorm).save(path)