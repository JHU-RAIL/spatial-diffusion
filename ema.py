import torch
import copy

# Taken from https://github.com/filipbasara0/simple-diffusion/blob/main/simple_diffusion/ema.py
class EMA:
    def __init__(
        self,
        model: torch.nn.Module,
        base_gamma: float,
        total_steps: int
    ) -> None:
        # Save instance of model being trained
        self.online_model = model

        # Make a copy of the model and switch off gradients
        self.ema_model = copy.deepcopy(self.online_model)
        self.ema_model.requires_grad_(False)

        # Set gamma (decay rate) and total number of training steps
        self.base_gamma = base_gamma
        self.total_steps = total_steps

    def update_params(self, gamma: float) -> None:
        # Switch off gradients when updating the EMA model
        with torch.no_grad():
            # Copy model weights and buffers to the EMA model
            valid_types = [torch.float, torch.float16]
            for o_param, t_param in self._get_params():
                if o_param.dtype in valid_types and t_param.dtype in valid_types:
                    t_param.data.lerp_(o_param.data, 1. - gamma)

            for o_buffer, t_buffer in self._get_buffers():
                if o_buffer.dtype in valid_types and t_buffer.dtype in valid_types:
                    t_buffer.data.lerp_(o_buffer.data, 1. - gamma)

    def _get_params(self):
        # Return paired weights of the training and EMA model
        return zip(self.online_model.parameters(),
                   self.ema_model.parameters())

    def _get_buffers(self):
        # Return paired buffers of the training and EMA model
        return zip(self.online_model.buffers(),
                   self.ema_model.buffers())
    
    def update_gamma(self, current_step: int) -> float:
        # Wrap current step and total steps into a floating point tensor
        curr = torch.tensor(current_step, dtype=torch.float32)
        total = torch.tensor(self.total_steps, dtype=torch.float32)

        # Cosine EMA schedule (increase from base_gamma to 1)
        tau = 1. - (1 - self.base_gamma) * (torch.cos(torch.pi * curr / total) + 1) / 2
        return tau.item()