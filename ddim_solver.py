# filename: ddim_solver.py

import torch
import numpy as np

class DDIMSolver:
    def __init__(self, alpha_cumprods, timesteps=1000, ddim_timesteps=50):
        # DDIM solver from the LCM paper
        self.alphas_cumprod = alpha_cumprods
        self.timesteps = timesteps
        self.ddim_timesteps = ddim_timesteps
        self.step_ratio = self.timesteps // self.ddim_timesteps

        ddim_timesteps_list = np.asarray(list(range(0, self.timesteps, self.step_ratio)))
        self.ddim_timesteps = torch.from_numpy(ddim_timesteps_list.round().astype(np.int64))

    def to(self, device):
        self.ddim_timesteps = self.ddim_timesteps.to(device)
        return self

    def ddim_step(self, pred_x0, pred_noise, timestep_index):
        # Get the original alpha values for the current and previous timesteps
        alpha_cumprod = torch.from_numpy(self.alphas_cumprod).to(pred_x0.device)
        
        prev_timestep = self.ddim_timesteps[timestep_index] - self.step_ratio
        
        # Ensure prev_timestep is not negative
        prev_timestep = torch.clamp(prev_timestep, 0)
        
        # Get current and previous alpha cumprods
        alpha_prod_t = alpha_cumprod[self.ddim_timesteps[timestep_index]]
        alpha_prod_t_prev = torch.where(
            prev_timestep >= 0,
            alpha_cumprod[prev_timestep],
            torch.tensor(1.0, device=pred_x0.device)
        )
        
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        
        # Correctly reshape alpha/beta tensors for broadcasting
        alpha_prod_t = alpha_prod_t.view(-1, 1, 1, 1)
        alpha_prod_t_prev = alpha_prod_t_prev.view(-1, 1, 1, 1)
        beta_prod_t = beta_prod_t.view(-1, 1, 1, 1)
        beta_prod_t_prev = beta_prod_t_prev.view(-1, 1, 1, 1)

        # DDIM step formula
        pred_dir_xt = (1. - alpha_prod_t_prev - beta_prod_t_prev).sqrt() * pred_noise
        x_prev = alpha_prod_t_prev.sqrt() * pred_x0 + pred_dir_xt
        
        return x_prev