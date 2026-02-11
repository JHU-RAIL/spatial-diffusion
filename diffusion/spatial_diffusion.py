import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Union, Optional
from tqdm import tqdm
from enum import Enum
import warnings

class Transformation(Enum):
    RIGID_BODY = 1
    AFFINE = 2

class Schedule(Enum):
    LINEAR = 1
    COSINE = 2

class SpatialDiffusion:
    def __init__(
        self,
        num_timesteps: int = 65,
        transf_std_range: Tuple[float, float] = (0.01, 0.07),
        transl_std_range: Tuple[float, float] = (0.5, 4.0),
        type: Transformation = Transformation.AFFINE,
        schedule: Schedule = Schedule.COSINE,
        interp: str = 'bilinear',
        device: Optional[torch.device] = None,
        verbose: bool = True,
        pbar_leave: bool = True,
        eps: float = 1e-8
    ) -> None:
        """
        An implementation of the diffusor for a Spatial Diffusion model 
        to perform 3D affine registration. The forward diffusion process 
        is characterized by sampling a series of affine transformation matrices 
        according to a linear standard deviation scheduler for both the 
        transformation and translation components. Random transformations 
        are in the form M = I + A, where A is sampled from a normal gaussian 
        distribution and must have ||A|| < 1 to guarentee invertibility.

        Contains helper functions for sampling random timesteps and affine 
        transformations for training, as well as predicting the full 
        reverse diffusion process during inferencing. Expects inputs of 
        dimension B x C x D x H x W during training and inputs of dimension 
        B x 2 x D x H x W when inferencing, registering the input from 
        channel 1 to the input from channel 0.
        """

        # Check that the number of timesteps is a positive number
        if num_timesteps <= 0:
            raise Exception('Number of diffusion timesteps must be > 0.')

        # Check that the standard deviation range is defined correctly
        if transf_std_range[0] < 0 or transf_std_range[1] < transf_std_range[0]:
            raise Exception('Standard deviation range must be specified in the format (min, max), where min is non-negative.')

        if transl_std_range[0] < 0 or transl_std_range[1] < transl_std_range[0]:
            raise Exception('Standard deviation range must be specified in the format (min, max), where min is non-negative.')

        self.num_timesteps = num_timesteps

        if schedule == Schedule.LINEAR:
            # Divide standard deviation range for transformation and translation magnitudes into 
            # regularly interspaced intervals
            self.transf_std_steps = torch.linspace(transf_std_range[0], transf_std_range[1], num_timesteps, device=device)
            self.transl_std_steps = torch.linspace(transl_std_range[0], transl_std_range[1], num_timesteps, device=device)
            print('\nSelected the LINEAR transformation schedule.')

        elif schedule == Schedule.COSINE:
            # Divide the standard deviation range for transformation and translation magnitudes 
            # into intermediate timesteps based on a cosine schedule
            cos_steps = 1. - torch.cos(0.5 * torch.pi * torch.linspace(0, 1, num_timesteps, device=device)) ** 2
            self.transf_std_steps = transf_std_range[0] + (transf_std_range[1] - transf_std_range[0]) * cos_steps
            self.transl_std_steps = transl_std_range[0] + (transl_std_range[1] - transl_std_range[0]) * cos_steps
            print('\nSelected the COSINE transformation schedule.')
        
        else:
            raise ValueError(f'Unknown transformation schedule \'{schedule}\'.')

        # Create a transformation and translation standard deviation schedule for each timestep
        self.transf_std_schedule = torch.sqrt(torch.cumsum(self.transf_std_steps ** 2, dim=0))
        self.transl_std_schedule = torch.sqrt(torch.cumsum(self.transl_std_steps ** 2, dim=0))

        self.type = type
        self.interp = interp
        self.device = device
        self.verbose = verbose
        self.pbar_leave = pbar_leave
        self.eps = eps

        # Display aggregated transformation and translation magnitude sampling mean and std
        std_transf = self.transf_std_schedule[-1].item()
        std_transl = self.transl_std_schedule[-1].item()

        if type == Transformation.RIGID_BODY:
            print('\nSpatial Diffusion Rigid-body Sampling Distributions:')
            print(f'Transformation: Mean = {0.0:.2f}, Std = {std_transf:.2f}')
            print(f'Translation: Mean = {0.0:.2f}, Std = {std_transl:.2f}\n')

        elif type == Transformation.AFFINE:
            print('\nSpatial Diffusion Affine Sampling Distributions:')
            print(f'Transformation: Mean = {0.0:.2f}, Std = {std_transf:.2f}')
            print(f'Translation: Mean = {0.0:.2f}, Std = {std_transl:.2f}\n')
            
        else:
            raise ValueError(f'Unknown transformation type \'{type}\'.')

        # Transformation magnitudes >= 1 can result in non-invertible matrices,
        # which will result in a runtime exception when torch.inverse() is called
        if std_transf >= 1:
            warnings.warn(f'Undefined behavior. The maximum transformation magnitude {std_transf:.2f} must be < 1 to guarentee invertibility!')
    
    def interp_mode(self, interp: str):
        """
        Set interpolation mode for applying transformations on volumes.
        """
        self.interp = interp

    def sample_at_timesteps(
        self,
        input: torch.Tensor,
        timesteps: torch.Tensor,
        trunc_sigma: Optional[float] = 2.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies a random transformation to a 3D volume with dimensions B x C x D x H x W.
        Transformation magnitude is sampled based on a Tensor of B different timesteps and
        the forward spatial diffusion transformation schedule. Expects input intensities to
        be in the range [0, 1] or [0, 255].
        """
        
        B = input.size(0)

        # Check that timestep is within the correct range
        if (timesteps < 1).any():
            raise Exception('Timesteps must be >= 1.')
        
        if (timesteps > self.num_timesteps).any():
            raise Exception(f'Timesteps must be <= initialized number of timesteps {self.num_timesteps}.')

        # Check that minimum intensity is non-negative
        if input.min() < 0:
            warnings.warn(f'The lower bounds of your intensity range {input.min():.1f} is negative. '
                          'Zero padding when transforming the volume may result in undesirable effects.')

        if self.type == Transformation.RIGID_BODY:
            # Randomly sample transformation angles (in radians) at current timestep from a normal distribution
            # Summing normal distributions yields a distribution that's the sum of variances
            std = self.transf_std_schedule[timesteps - 1]
            transf = torch.randn((3, B), device=self.device) * std

            # Randomly sample transformation log-scale factor at current timestep from a normal distribution 
            # using the same schedule as the transformation angles
            scale = torch.randn((3, B), device=self.device) * std

            # Truncate any extreme transformation matrix rotations and log-scale factors > trunc_sigma * std
            if trunc_sigma is not None:
                bounds = trunc_sigma * std
                transf = torch.clamp(transf, min=-bounds, max=bounds)
                scale = torch.clamp(scale, min=-bounds, max=bounds)
            
            # Exponentiate the log-scale factors
            scale = torch.exp(scale)

        elif self.type == Transformation.AFFINE:
            # Randomly sample transformation magnitudes at current timestep from normal distribution
            # Summing normal distributions yields a distribution that's the sum of variances
            std = self.transf_std_schedule[timesteps - 1]
            transf = torch.randn(B, device=self.device) * std

            # Truncate any extreme transformation matrix magnitude > trunc_sigma * std
            if trunc_sigma is not None:
                bounds = trunc_sigma * std
                transf = torch.clamp(transf, min=-bounds, max=bounds)

            # No need to sample transformation scale factor, since it is an implicit property of 
            # the transformation magnitude
            scale = None

            # Transformation is not invertible if the magnitude >= 1 or <= -1
            # Raises a warning when transformation magnitude exceeds these bounds
            if (transf.abs() >= 1).any():
                warnings.warn('Invertibility cannot be guarenteed since transformation magnitude is >= 1. May result in undefined behavior.')
        
        else:
            raise Exception('Unknown transformation type. Must be either ' \
                            '\'Transformation.RIGID_BODY\' or \'Transformation.AFFINE\'.')

        # Randomly sample translations at current timestep from normal distribution
        # Summing normal distributions yields a distribution that's the sum of variances
        std = self.transl_std_schedule[timesteps - 1][None,]
        transl = torch.randn((3, B), device=self.device) * std

        # Truncate any extreme translations > trunc_sigma * std
        if trunc_sigma is not None:
            bounds = trunc_sigma * std
            transl = torch.clamp(transl, min=-bounds, max=bounds)

        # Apply the forward diffusion process transformation for the current timestep
        q_t, _, affine_mat = self._compute_transformations(
            input, transf, transl, scale, self.type, apply_transf=True
        )

        # Compute the reverse diffusion affine transformation at the current timestep
        # Invert the transformation matrix
        inv_transf = torch.inverse(affine_mat[:,:3,:3])

        # Compute the inverse's translation vector
        inv_transl = -inv_transf @ affine_mat[:,:3,3].view(-1, 3, 1)

        # Concatenate transformation and translation components of inverse affine matrix
        inv_affine_mat = torch.cat([inv_transf, inv_transl], dim=2)
        return q_t, inv_affine_mat

    def sample_timesteps(self, n: int, device: torch.device = None) -> torch.Tensor:
        """
        Randomly sample a diffusion timestep from a uniform distribution.
        """
        return torch.randint(low=1, high=(self.num_timesteps + 1), size=(n,), device=device)

    def predict(
        self,
        input: torch.Tensor,
        model: nn.Module,
        prepr_type: Union[str, Tuple[str, str]] = 'none',
        class_cond: Optional[torch.Tensor] = None,
        save_steps: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts the spatial diffusion model on a volume pair. Expects an input 
        tensor of dimension B x 2 x D x H x W. Returns the volume from input 
        channel 1 registered to input channel 0. The parameter save_steps
        determines whether to save every intermediate output for each diffusion
        step and return them as a list.
        """

        B = input.size(0)

        # Check that input contains fixed reference and transforming (moving) input
        if input.size(1) != 2:
            raise Exception('Input must contain two channels: A fixed reference as the first index and transforming as the second index.')
        
        # Set model to evaluation mode
        model.eval()
        output = None

        # Initialize a list storing outputs for intermediate diffusion steps
        if save_steps:
            intermediates = []

        with torch.inference_mode():
            # Create tensor of updated reverse diffusion outputs and composed transformation
            # input[:,0,:,:,:] is the fixed reference and input[:,1,:,:,:] is being transformed (moving)
            output = input.clone()
            composed_transf = torch.eye(4, device=self.device).view(1, 4, 4).repeat(B, 1, 1)
            I = torch.eye(4, device=self.device).view(1, 4, 4).repeat(B, 1, 1)
            
            for i in tqdm(reversed(range(1, self.num_timesteps + 1)), desc='Registering Volumes',
                          disable=not self.verbose, leave=not self.pbar_leave):
                # Create tensor of timesteps
                timestep = (torch.ones(B, device=self.device) * i).int()

                # Predict the model at timestep i
                affine_matrix = model(output, timestep, prepr_type, class_cond)[1]

                # Apply padding to 3x4 affine transformation matrix to create a 4x4 matrix
                pad = torch.zeros((B, 1, 4), device=self.device)
                pad[:,:,3] = 1.0
                timestep_transf = torch.cat([affine_matrix, pad], dim=1)

                # Calculate the transformation and translation magnitude standard deviation 
                # ratio for reverse diffusion at the current timestep
                alpha_transf = self.transf_std_steps[i - 1] / self.transf_std_schedule[i - 1]
                alpha_transl = self.transl_std_steps[i - 1] / self.transl_std_schedule[i - 1]

                # Compute the affine transformation for the reverse diffusion step
                timestep_transf[:,:3,:3] = (1. - alpha_transf) * I[:,:3,:3] + alpha_transf * timestep_transf[:,:3,:3]
                timestep_transf[:,:3,3] = (1. - alpha_transl) * I[:,:3,3] + alpha_transl * timestep_transf[:,:3,3]

                # Update the composed transformation across all timesteps up to i
                composed_transf = composed_transf @ timestep_transf

                # Apply the composed transformation of all timesteps up to i
                grid = F.affine_grid(composed_transf[:,:3], input.size(), align_corners=False)
                output = F.grid_sample(
                    model.unnormalize(input[:,1,:,:,:].unsqueeze(1)), grid, mode=self.interp,
                    padding_mode='zeros', align_corners=False
                )
                output = model.normalize(output)

                # Save intermediate states for the reverse diffusion process if necessary
                if save_steps:
                    intermediates.append(output.clone().detach().cpu().numpy())
                
                output = torch.cat([input[:,0,:,:,:].unsqueeze(1), output], dim=1)

        if save_steps:
            return output[:,1].unsqueeze(1), composed_transf[:,:3], intermediates
        else:
            return output[:,1].unsqueeze(1), composed_transf[:,:3]

    def predict_bidirectional(
        self,
        input_ab: torch.Tensor,
        model: nn.Module,
        prepr_type: Union[str, Tuple[str, str]] = 'none',
        class_cond: Optional[torch.Tensor] = None,
        save_steps: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predicts the spatial diffusion model on a volume pair bidirectionally. Given a 
        reference volume A (input channel 0) and transforming volume B (input channel 0), 
        the model first registers B --> A, outputting a transformed B'. Next, the model
        predicts A --> B' and the transformation is inverted (if possible) to get B' --> A.
        Output is the composed B --> A and B' --> A transformation.

        Expects an input tensor of dimension B x 2 x D x H x W. Returns the volume from 
        input channel 1 registered to input channel 0. The parameter save_steps determines
        whether to save every intermediate output for each diffusion step for both the
        B --> A and B' --> A transformations, then return them as a list.
        """

        # Predict the spatial diffusion model to register volume B --> A (original reference)
        if save_steps:
            unidir_reg_vols, transf_ba, steps_ba = self.predict(input_ab, model, prepr_type, class_cond, True)
        else:
            unidir_reg_vols, transf_ba = self.predict(input_ab, model, prepr_type, class_cond, False)

        # Initialize new tensor of volumes with the reference and transforming volumes switched
        input_ba = torch.zeros_like(input_ab, device=self.device)
        input_ba[:,0] = unidir_reg_vols[:,0]
        input_ba[:,1] = input_ab[:,0]

        # Transpose the specified preprocessing types as well if necessary
        if isinstance(prepr_type, Tuple):
            prepr_type = (prepr_type[1], prepr_type[0])

        # Predict the spatial diffusion model to transform volume A (original reference) --> B',
        # where B' is the transformed from volume registered from B --> A in the previous step
        if save_steps:
            registered_vols, transf_ab, steps_ab = self.predict(input_ba, model, prepr_type, class_cond, True)
        else:
            registered_vols, transf_ab = self.predict(input_ba, model, prepr_type, class_cond, False)

        # Convert B --> A affine transformation matrix into homogeneous coordinates
        BA = torch.eye(4, device=self.device).view(1, 4, 4).repeat(input_ab.size(0), 1, 1)
        BA[:,:3,:] = transf_ba

        # Convert A --> B' affine transformation matrix into homogeneous coordinates
        AB = torch.eye(4, device=self.device).view(1, 4, 4).repeat(input_ba.size(0), 1, 1)
        AB[:,:3,:] = transf_ab

        # Compute to determinant
        det = torch.linalg.det(AB)

        # Invert the A --> B' transformation to get B' --> A (if possible)
        AB_inv = torch.zeros_like(AB, device=self.device)
        AB_inv[det == 0] = torch.eye(4, device=self.device)
        AB_inv[det != 0] = torch.linalg.inv(AB[det != 0])

        # Compose the B --> A and B' --> A transformations
        composed_transf = (BA @ AB_inv)[:,:3]

        # Apply composed transformation to the input
        grid = F.affine_grid(composed_transf, input_ab.size(), align_corners=False)
        output = F.grid_sample(
            model.unnormalize(input_ab[:,1,:,:,:].unsqueeze(1)), grid, mode=self.interp,
            padding_mode='zeros', align_corners=False
        )
        output = model.normalize(output)

        # Return both bidirectional and unidirectional registration results
        if save_steps:
            return (output, composed_transf, steps_ab), (unidir_reg_vols, transf_ba, steps_ba)
        else:
            return (output, composed_transf), (unidir_reg_vols, transf_ba)

    def random_transf_predict(
        self,
        input: torch.Tensor,
        model: nn.Module,
        prepr_type: Union[str, Tuple[str, str]] = 'none',
        class_cond: Optional[torch.Tensor] = None,
        transf_magnitude: Union[Tuple[float, float], float] = (-0.35, 0.35),
        translation: Union[Tuple[float, float], float] = (-20.0, 20.0)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Applies a random transformation on the input volume of dimension B x 1 x D x H x W
        and predicts reverse diffusion process for registration.
        """
        
        # Check that input contains only a fixed reference input
        if input.size(1) != 1:
            raise Exception('Input must contain one channels: A fixed reference that will be randomly transformed.')

        # If a scalar transformation magnitude is specified, default to a 
        # range of (-|transf_magnitude|, |transf_magnitude|)
        if not isinstance(transf_magnitude, Tuple):
            transf_magnitude = (-abs(transf_magnitude), abs(transf_magnitude))

        # If a scalar translation is specified, default to a 
        # range of (-|translation|, |translation|)
        if not isinstance(translation, Tuple):
            translation = (-abs(translation), abs(translation))

        # Randomly select translation
        rand_transl = (translation[0] + (translation[1] - translation[0]) * torch.rand((3, input.size(0)), device=self.device))

        if self.type == Transformation.RIGID_BODY:
            # Randomly select transformation matrix angles (in radians)
            rand_angles = (transf_magnitude[0] + (transf_magnitude[1] - transf_magnitude[0]) * 
                           torch.rand((3, input.size(0)), device=self.device))

            # Randomly select transformation matrix scale factor in log space
            rand_scale = torch.exp(transf_magnitude[0] + (transf_magnitude[1] - transf_magnitude[0]) * 
                                   torch.rand((3, input.size(0)), device=self.device))
            
            # Compute rigid-body transformation matrix
            rand_transf = torch.stack([self._rigid_matrix((rand_angles[0,i], rand_angles[1,i], rand_angles[2,i]),
                                       (rand_transl[0,i], rand_transl[1,i], rand_transl[2,i]),
                                       (rand_scale[0,i], rand_scale[1,i], rand_scale[2,i]),
                                       input.shape[2:]) for i in range(rand_angles.size(1))])

        elif self.type == Transformation.AFFINE:
            # Randomly select transformation matrix magnitude
            rand_mag = (transf_magnitude[0] + (transf_magnitude[1] - transf_magnitude[0]) * 
                        torch.rand(input.size(0), device=self.device))

            # Transformation is not invertible if the magnitude >= 1 or <= -1
            # Raises a warning when transformation magnitude exceeds these bounds
            if (rand_mag.abs() >= 1).any():
                warnings.warn('Invertibility cannot be guarenteed since transformation magnitude is >= 1. May result in undefined behavior.')

            # Compute affine transformation matrix
            rand_transf = torch.stack([self._random_affine_matrix(rand_mag[i], (rand_transl[0,i], 
                                    rand_transl[1,i], rand_transl[2,i]), input.shape[2:])
                                    for i in range(rand_mag.size(0))])
        
        else:
            raise Exception('Unknown transformation type. Must be either ' \
                            '\'Transformation.RIGID_BODY\' or \'Transformation.AFFINE\'.')

        # Apply rotation onto the input
        rand_grid = F.affine_grid(rand_transf, input.size(), align_corners=False)
        input_transf = F.grid_sample(
            model.unnormalize(input), rand_grid, mode=self.interp,
            padding_mode='zeros', align_corners=False
        )
        input_transf = model.normalize(input_transf)

        # Predict the reverse affine transformation
        output, composed_transf = self.predict(
            torch.cat([input, input_transf], dim=1),
            model, prepr_type, class_cond
        )

        return output, input_transf, composed_transf
    
    def _compute_transformations(
        self,
        input: torch.Tensor,
        transf_magn: torch.Tensor,
        transl_magn: torch.Tensor,
        scale_magn: Optional[torch.Tensor] = None,
        type: Transformation = Transformation.AFFINE,
        apply_transf: bool = True
    ) -> torch.Tensor:
        """
        Compute transformations and apply onto input volume of dimension B x 1 x D x H x W.
        """
        
        if type == Transformation.RIGID_BODY:
            # Compute rigid-body transformation matrices for the current timestep
            transf_matrix = torch.stack([self._rigid_matrix((transf_magn[0,i], transf_magn[1,i], transf_magn[2,i]), 
                                         (transl_magn[0,i], transl_magn[1,i], transl_magn[2,i]), 
                                         None if scale_magn is None else (scale_magn[0,i], scale_magn[1,i],
                                         scale_magn[2,i]), input.shape[2:]) for i in range(transf_magn.size(1))])
        
        elif type == Transformation.AFFINE:
            # Scale factor will be ignored when generator random affine matrices, since it is an inherent
            # property of the transformation magnitude
            if scale_magn is not None:
                warnings.warn('Scaling magnitude will be ignored when generating random affine matrices!')
            
            # Compute affine transformation matrices for the current timestep
            transf_matrix = torch.stack([self._random_affine_matrix(transf_magn[i], (transl_magn[0,i], 
                                         transl_magn[1,i], transl_magn[2,i]), input.shape[2:]) 
                                         for i in range(transf_magn.size(0))])
        
        else:
            raise Exception('Unknown transformation type. Must be either ' \
                            '\'Transformation.RIGID_BODY\' or \'Transformation.AFFINE\'.')

        # Compute the transformation
        grid = F.affine_grid(transf_matrix, input.size(), align_corners=False)

        if apply_transf:
            # Apply transformation on the input
            transformed = F.grid_sample(
                input, grid, mode=self.interp, padding_mode='zeros', align_corners=False
            )

            return transformed, grid, transf_matrix
        else:
            return grid, transf_matrix
    
    def _rigid_matrix(
        self,
        angle: Union[Tuple[torch.Tensor, ...], torch.Tensor],
        translation: Union[Tuple[torch.Tensor, ...], torch.Tensor],
        scale: Optional[Union[Tuple[torch.Tensor, ...], torch.Tensor]] = None,
        dims: Tuple[int, int, int] = None
    ) -> torch.Tensor:
        """
        Create a 3D rigid-body transformation matrix.
        """

        # If a scalar angle is specified, default to using same angle in each axis
        if not isinstance(angle, Tuple):
            angle = (angle, angle, angle)

        # If a scalar translation is specified, default to using same translation in each axis
        if not isinstance(translation, Tuple):
            translation = (translation, translation, translation)

        # If a scalar scale factor is specified, default to using same scale factor in each axis
        if scale is None:
            scale = (torch.tensor(1.0, device=angle.device), torch.tensor(1.0, device=angle.device), 
                     torch.tensor(1.0, device=angle.device))
        elif not isinstance(scale, Tuple):
            scale = (scale, scale, scale)

        # Define rotation matrix around the x-axis
        R_x = torch.tensor([[1.0, 0.0, 0.0],
                            [0.0, torch.cos(angle[0]), -torch.sin(angle[0])],
                            [0.0, torch.sin(angle[0]), torch.cos(angle[0])]], device=self.device)
    
        # Define rotation matrix around the y-axis
        R_y = torch.tensor([[torch.cos(angle[1]), 0.0, torch.sin(angle[1])],
                            [0.0, 1.0, 0.0],
                            [-torch.sin(angle[1]), 0.0, torch.cos(angle[1])]], device=self.device)
        
        # Define rotation matrix around the z-axis
        R_z = torch.tensor([[torch.cos(angle[2]), -torch.sin(angle[2]), 0],
                            [torch.sin(angle[2]), torch.cos(angle[2]), 0],
                            [0.0, 0.0, 1.0]], device=self.device)

        # Translation vector in the x, y, and z-axis
        transl = torch.tensor([
            [translation[0] / (dims[0] / 2.0)],
            [translation[1] / (dims[1] / 2.0)],
            [translation[2] / (dims[2] / 2.0)]
        ], device=self.device)

        # Combined rotation matrix
        R = R_x @ R_y @ R_z

        # Apply scaling after rotation
        S = torch.diag(torch.tensor([scale[0], scale[1], scale[2]], device=self.device))
        RS = R @ S

        # Combine rotation and translation components to create a rigid-body transformation matrix
        transf_matrix = torch.cat([RS, transl], dim=1)
        return transf_matrix

    def _random_affine_matrix(
        self,
        transf_magnitude: torch.Tensor,
        translation: Union[Tuple[torch.Tensor, ...], torch.Tensor],
        dims: Tuple[int, int, int]
    ) -> torch.Tensor:
        """
        Generate a random, invertible 3D affine transformation matrix
        in the form I + A, where A is a random matrix sampled from a
        normal distribution with a spectral norm of the specified 
        transformation magnitude. The norm of the matrix ||A|| < 1 must
        be satisfied to guarentee invertibility.
        """

        # Create identity affine transformation matrix
        I = torch.eye(4, device=self.device)[:3]

        # If a scalar translation is specified, default to using same translation in each axis
        if not isinstance(translation, Tuple):
            translation = (translation, translation, translation)

        # Sample a random transformation matrix from a normal distribution and scale the matrix
        # so that the spectral norm matches the transformation magnitude
        transform = torch.randn(3, 3, device=self.device)
        transform *= transf_magnitude / (torch.linalg.norm(transform, 2) + self.eps)

        # Set the translation components of the affine transformation
        aff = torch.zeros((3, 1), device=self.device)

        for i in range(3):
            aff[i,0] = translation[i] / (dims[i] / 2.0)

        # Concatinate random transformation and translation components to create the 
        # purturbation from the identity matrix
        perturbation = torch.cat([transform, aff], dim=1)

        # Apply the perturbation and generate the random affine transformation matrix
        transf_matrix = I + perturbation
        return transf_matrix