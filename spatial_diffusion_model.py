import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Tuple, Union, Optional, Callable
    
class AffineTransformer(nn.Module):
    def __init__(
        self,
        input_range: Tuple[float, float] = (-1.0, 1.0),
        input_dim: Optional[Union[int, Tuple[int, int, int]]] = None,
        channels_enc: Tuple[int, ...] = (2, 64, 128, 256, 256, 512, 1024),
        time_dim: int = 256,
        time_mlp_ratio: int = 4,
        n_classes: int = 1,
        in_groups: Optional[int] = 32,
        out_groups: Optional[int] = 32,
        activation: Callable[[torch.Tensor], torch.Tensor] = nn.SiLU,
        output_kernel: Union[int, Tuple[int, int, int]] = (4, 6, 4),
        class_dropout: Optional[float] = 0.15,
        interp: str = 'bilinear'
    ) -> None:
        """
        An Affine Transformer model for spatial diffusion, where given an 
        input volume of dimension B x 2 x D x H x W, estimates the 
        affine transformation matrix to register the input at channel 1 
        to the input at channel 0. The model uses a pre-activation ResNet 
        encoder architecture.

        The transformation is predicted at a given timestep, which is given
        as a B-dimensional vector. Time embeddings are initially represented
        using sinusoidal positional embeddings, then non-linearly transformed
        and encorporated into every ResNet block of the encoder. Stage-specific
        modality-conditional embeddings for the reference and transforming volumes
        are added to the time embeddings.
        """
        super().__init__()

        # Input range should be a tuple containing (min, max) intensities
        if not isinstance(input_range, Tuple) or len(input_range) != 2:
            raise ValueError('Input range must be a tuple containing (min, max) intensities!')

        # For input range (min, max), the max intensity must be greater than the min
        if input_range[0] >= input_range[1]:
            raise ValueError('Minimum input range intensity must be less than the maximum!')

        # Number of modality-conditional classes must be > 0
        if n_classes <= 0:
            raise ValueError('Number of modality-conditional classes must be > 0! ' \
                'Use n_classes = 1 for unconditional training.')

        # Set model input volume size -- all inputs will be resized to specified dimensions
        if input_dim is not None and not isinstance(input_dim, Tuple):
            self.input_dim = (input_dim, input_dim, input_dim)
        else:
            self.input_dim = input_dim

        self.input_range = input_range
        self.time_dim = time_dim
        self.time_mlp_ratio = time_mlp_ratio
        self.n_classes = n_classes
        self.interp = interp

        # Initialize MLP network for transforming time embeddings
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim)
        )

        # Initialize encoder network
        self.encoder = Encoder(
            channels_enc, time_dim, n_classes, in_groups, out_groups, 
            activation, class_dropout
        )

        # Set and/or check output groups
        if out_groups is None:
            out_groups = min(32, channels_enc[-1] // 4)
        elif out_groups > channels_enc[-1]:
            raise ValueError(f'Number of output channels ({out_channels}) '\
                f'must be >= number of output groups ({out_groups})!')
        
        # Initialize output head
        self.output_head = nn.Sequential(
            nn.GroupNorm(out_groups, channels_enc[-1]),
            activation(),
            nn.Conv3d(channels_enc[-1], 12, kernel_size=output_kernel, stride=1, padding=0)
        )

    def forward(
        self,
        input: torch.Tensor,
        timestep: torch.Tensor,
        prepr_type: Union[str, Tuple[str, str]] = 'none',
        cond_class: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the Affine Transformer model.
        Expects an input of dimension B x 2 x D x H x W.
        Outputs the transformed input at channel 1 registered
        to the input at channel 0 along with the affine matrix.

        The timestep is a B-dimensional vector of integers 
        representing the temporal position within the diffusion 
        process. The string prepr_type represents the preprocessing 
        strategy to apply  on the input. Modality-conditional classes are 
        represented as B-dimensional vector.
        """

        input_dim = tuple(input.shape[2:])

        # Resize the input volume to target input dimension if necessary
        if self.input_dim is not None and self.input_dim != input_dim:
            x = F.interpolate(input, size=self.input_dim, mode='trilinear', align_corners=False).detach()
            scaling = torch.tensor([[
                input_dim[2] / self.input_dim[2],
                input_dim[1] / self.input_dim[1],
                input_dim[0] / self.input_dim[0]
            ]], device=input.device)
        else:
            x = input
            scaling = torch.ones((1, 3), device=input.device)
        
        # Preprocessing type must be a string or tuple of length 2
        if isinstance(prepr_type, Tuple) and len(prepr_type) != 2:
            raise ValueError('Preprocessing type must be a string or a tuple of strings with length 2!')
        elif not isinstance(prepr_type, Tuple):
            prepr_type = (prepr_type, prepr_type)

        # Create identity affine matrix
        I = torch.eye(4, device=input.device)[None,:3]

        # Preprocess the input volumes
        if prepr_type[0] == prepr_type[1]:
            # Preprocess both the reference and transformed volumes in the same way
            prepr = self._preprocess(x, prepr_type[0])
        else:
            # Preprocess the reference and transformed volumes separately
            prepr_ref = self._preprocess(x[:,0][:,None,], prepr_type[0])
            prepr_transf = self._preprocess(x[:,1][:,None,], prepr_type[1])

            # Concatenate the preprocessed reference and transformed volumes
            prepr = torch.cat([prepr_ref, prepr_transf], dim=1)

        # Compute time embeddings
        timestep = timestep.type(torch.float)
        t_emb = self._sinusoidal_pos_encoding(timestep, self.time_dim)
        t_emb = self.time_mlp(t_emb)
        
        # Encode input and generate affine transformation matrix
        features = self.encoder(prepr, t_emb, cond_class)
        affine = I + self.output_head(features).view(-1, 3, 4)
        affine[:,:3,3] = affine[:,:3,3] * scaling

        # Apply affine transformation
        transformed = self._apply_transf(input[:,1].unsqueeze(1), affine)
        assert input.shape[2:] == transformed.shape[2:], \
            f'Input shape {input.shape[2:]} should match output shape {transformed.shape[2:]}!'
        
        return transformed, affine
    
    def normalize(self, input: torch.Tensor) -> torch.Tensor:
        """
        Normalize the input with range (0, 1) to (min, max).
        """

        return input * (self.input_range[1] - self.input_range[0]) + self.input_range[0]

    def unnormalize(self, input: torch.Tensor) -> torch.Tensor:
        """
        Un-normalize the input with range (min, max) to (0, 1).
        """

        return (input - self.input_range[0]) / (self.input_range[1] - self.input_range[0])
        
    def _preprocess(self, input: torch.Tensor, type: Union[str, Tuple[str, str]] = 'none') -> torch.Tensor:
        """
        Preprocess the input volume prior to input into the model
        by applying Gabor filters oriented about the x and y axis,
        then thresholding the volume.

        The string type represents the preprocessing strategy to use.
        Choices are 'oct' for retinal OCT volumes, 'octa' for retinal
        OCTA volumes, 'mri' for brain MRI volumes, 'venous' for venous
        malformation MRI volumes, or 'none' to apply no preprocessing.
        """

        # Un-normalize input into the range [0, 1]
        input = self.unnormalize(input)

        if type.lower() == 'oct':
            # Set Gabor filter and threshold parameters for retinal OCT data
            omega = 0.5
            threshold = -0.4

        elif type.lower() == 'octa':
            # Set Gabor filter and threshold parameters for retinal OCTA data
            omega = 0.95
            threshold = -0.5

        elif type.lower() == 'mri' or type.lower() == 'venous' or type.lower() == 'none':
            # Don't apply any preprocessing -- re-normalize and reset gradients to its original setting
            input = self.normalize(input)
            return input
        
        else:
            # Encountered unexpected data type
            raise ValueError(f'Unknown preprocessing type. Expected \'oct\', \'octa\', '\
                f'\'mri\', \'venous\', or \'none\', but got \'{type}\'!')

        if omega is None:
            # An omega is not provided - don't apply gabor filters
            prepr = input
        else:
            # Create Gabor filters oriented in the x and y direction
            gabor_filter_x = GaborFilter3d(
                kernel_size=5, theta=torch.pi/2, phi=0.0, omega=omega,
                type='cosine', device=input.device
            )

            gabor_filter_y = GaborFilter3d(
                kernel_size=5, theta=0.0, phi=0.0, omega=omega,
                type='cosine', device=input.device
            )

            # Apply Gabor filter
            prepr = torch.stack([gabor_filter_x(input), gabor_filter_y(input)], dim=0)
            prepr = self.normalize(prepr)   # Re-normalize output into the input intensity range
            prepr = torch.max(prepr, dim=0)[0]

        # Threshold the voxel intensities
        prepr = (prepr >= threshold).float()
        
        # Re-normalize output
        return self.normalize(prepr)

    def _sinusoidal_pos_encoding(
        self,
        timestep: torch.Tensor,
        channels: int
    ) -> torch.Tensor:
        """
        Uses a sinusoidal positional encoding scheme to 
        generate time embeddings.
        """
        device = timestep.device
        half_dim = channels // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = timestep[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

    def _apply_transf(self, input: torch.Tensor, affine: torch.Tensor) -> torch.Tensor:
        """
        Apply an affine transformation onto an input volume.
        """

        # Generate affine transformation grid given a matrix
        grid = F.affine_grid(affine, input.size(), align_corners=False)

        # Apply affine transformation on the volume (NOTE: Always use 
        # the differentiable linear interpolation for training)
        transformed = F.grid_sample(
            self.unnormalize(input), grid, mode=self.interp,
            padding_mode='zeros', align_corners=False
        )
        transformed = self.normalize(transformed)

        return transformed

class Encoder(nn.Module):
    def __init__(
        self,
        channels_enc: Tuple[int, ...] = (2, 64, 128, 256, 256, 512, 1024),
        time_dim: int = 256,
        n_classes: int = 1,
        in_groups: Optional[int] = 32,
        out_groups: Optional[int] = 32,
        activation: Callable[[torch.Tensor], torch.Tensor] = nn.SiLU,
        class_dropout: Optional[float] = 0.15
    ) -> None:
        """
        Basic encoder network for the spatial diffusion Affine Transformer
        model. The architecture follows a pre-activation ResNet design 
        with group normalization, time embeddings, and modality-conditional
        embeddings.
        """

        super().__init__()

        # Input convolution layer
        self.in_conv = nn.Conv3d(channels_enc[0], channels_enc[1], kernel_size=4, stride=2, padding=1)

        # Encoder layers
        self.encoder = nn.ModuleList(
            [DownBlock(channels_enc[i], channels_enc[i+1], time_dim, n_classes, in_groups, out_groups,
             activation, class_dropout) for i in range(1, len(channels_enc) - 1)]
        )

    def forward(
        self, input: torch.Tensor,
        time: torch.Tensor,
        cond_class: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through encoder network. Expects an input 
        of dimension B x C x D x H x W, time embeddings of 
        dimension B x T, and modality-conditional classes as a 
        B-dimensional vector.
        """

        # Pass through input convolution
        x = self.in_conv(input)

        # Forward pass through encoder downsampling ResNet blocks
        for encoder_block in self.encoder:
            x = encoder_block(x, time, cond_class)
        
        return x
    
class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        n_classes: int = 1,
        in_groups: Optional[int] = 32,
        out_groups: Optional[int] = 32,
        activation: Callable[[torch.Tensor], torch.Tensor] = nn.SiLU,
        class_dropout: Optional[float] = 0.15
    ) -> None:
        """
        Downsampling encoder block for the spatial diffusion 
        Affine Transformer model. Block follows a pre-activation
        ResNet design with group normalization, time embeddings,
        and modality-conditional embeddings for the reference and
        transforming volumes.
        """

        super().__init__()

        # Class dropout probability must be in the range [0, 1]
        if class_dropout < 0.0 or class_dropout > 1.0:
            raise ValueError(f'Class dropout probability {class_dropout:.2f} must be within the range [0, 1].')

        if in_groups is None:
            in_groups = min(32, in_channels // 4)
        elif in_groups > in_channels:
            raise ValueError(f'Number of input channels ({in_channels}) must be >= number of input groups ({in_groups})!')
        
        if out_groups is None:
            out_groups = min(32, out_channels // 4)
        elif out_groups > out_channels:
            raise ValueError(f'Number of output channels ({out_channels}) must be >= number of output groups ({out_groups})!')

        # Modality-conditional class dropout probability
        self.class_dropout = class_dropout

        # First convolution
        self.conv1 = nn.Sequential(
            nn.GroupNorm(in_groups, in_channels),
            activation(),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        )

        # Second convolution
        self.conv2 = nn.Sequential(
            nn.GroupNorm(out_groups, out_channels),
            activation(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )

        # Skip connection convolution for spatial and channel resizing
        self.res_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2, ceil_mode=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        )

        # Projects time embeddings
        self.time_embed = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_dim, out_channels)
        ) if time_dim is not None else None

        # Creates class embeddings
        self.class_embedding = nn.Embedding(
            n_classes + 1 if self.class_dropout else n_classes, out_channels
        ) if n_classes > 1 else None

    def forward(
        self, input: torch.Tensor,
        time: torch.Tensor,
        cond_class: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through Downsampling encoder block.
        Expects an input of dimension B x C x D x H x W,
        time embeddings of dimension B x T, and modality-
        conditional classes as a B x 2 vector, one for the 
        reference and transforming volumes.
        """

        if not self.class_embedding and cond_class is not None:
            raise ValueError('Modality-conditional class provided, but model is not initialized ' \
                             'for modality-conditional registration!')
        
        elif self.class_embedding and cond_class is None:
            raise ValueError('Model is initialized for modality-conditional registration, ' \
                             'but no modality-conditional class was provided!')

        if self.time_embed is not None and time.size(0) != input.size(0):
            raise ValueError(f'Timestep batch size of {time.size(0)} must match input' \
                             f'batch size of {input.size(0)}!')

        if self.class_embedding and cond_class.size(0) != input.size(0):
            raise ValueError(f'Modality-conditional class batch size of {cond_class.size(0)} must ' \
                             f'match input batch size of {input.size(0)}!')
        
        if self.class_embedding and len(cond_class.shape) != 1:
            raise ValueError('Modality-conditional class must be a B-dimensional vector, but got a ' \
                             f'{len(cond_class.shape)}-dimensional tensor instead!')

        if self.class_embedding and not ((cond_class >= 0) & (cond_class < self.class_embedding.weight.size(0))).all():
            raise ValueError('Modality-conditional class for must be an integer within the range ' \
                             f'[0, {self.class_embedding.weight.size(0) - 1}] based on the model\'s configuration!')

        if self.training and self.class_dropout and self.class_embedding and (cond_class == 0).any():
            raise ValueError('Modality-conditional class 0 is reserved during training when class dropout is enabled ' \
                             'and should not be explicitly passed in or used as part of cond_class.')

        # Pass through first convolution
        x = self.conv1(input)

        # Add time embedding to feature maps
        if self.time_embed is not None:
            emb = self.time_embed(time)

            # Add class embeddings if necessary
            if self.class_embedding is not None:
                # Determine class dropout if necessary
                if self.training and self.class_dropout:
                    dropout = torch.rand_like(cond_class, dtype=torch.float, device=cond_class.device)
                else:
                    dropout = torch.ones_like(cond_class, dtype=torch.float, device=cond_class.device)
                
                # Integrate class embeddings
                emb = emb + self.class_embedding(cond_class * (dropout >= self.class_dropout))

            x = x + emb[:,:,None,None,None]

        # Pass through second convolution
        x = self.conv2(x)
        
        # Add residual connection
        return x + self.res_conv(input)

class GaborFilter3d(nn.Module):
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int, int]] = 5,
        theta: float = 0.0,
        phi: float = 0.0,
        omega: float = 0.95,
        k: torch.Tensor = torch.pi,
        type: str = 'cosine',
        device: torch.device = None
    ) -> None:
        """
        Create and apply a Gabor filter on 3D volumes.
        """
        
        super().__init__()

        # Kernel size of the gabor filter
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        # Rotation angle of Gabor filter in the xy-plane (theta) and xz-plane (phi)
        self.theta = torch.tensor(theta, device=device)
        self.phi = torch.tensor(phi, device=device)

        # Frequency, scaling, and sinusoidal function parameters for the Gabor filter
        self.omega = torch.tensor(omega, device=device)
        self.k = torch.tensor(k, device=device)
        self.filter = type

        # Set PyTorch tensor device
        self.device = device
    
    def create_gabor_kernel(self) -> torch.Tensor:
        """
        Create a 3D Gabor filter kernel.
        """
        
        # Kernel size must be 3-dimensional
        assert len(self.kernel_size) == 3, "Size must be a tuple of 3 integers (D, H, W)"
    
        d, h, w = self.kernel_size
        z, y, x = torch.meshgrid(
            torch.arange(-(d // 2), d // 2 + 1, device=self.device),
            torch.arange(-(h // 2), h // 2 + 1, device=self.device),
            torch.arange(-(w // 2), w // 2 + 1, device=self.device),
            indexing='ij'
        )
        
        # Rotate the grid for orientation in xy-plane (theta)
        x_theta = x * torch.cos(self.theta) + y * torch.sin(self.theta)
        y_theta = -x * torch.sin(self.theta) + y * torch.cos(self.theta)
        
        # Rotate the grid for orientation in xz-plane (phi)
        x_phi = x_theta * torch.cos(self.phi) + z * torch.sin(self.phi)
        z_phi = z * torch.cos(self.phi) - x_theta * torch.sin(self.phi)
        
        # Create the Gabor kernel
        gaussian = (self.omega ** 2 / (4 * torch.pi * self.k ** 2)) * \
                    torch.exp(-self.omega ** 2 / (8 * self.k ** 2) * \
                    (4 * x_phi ** 2 + y_theta ** 2 + z_phi ** 2))

        # Create sinusoidal function for Gabor filter (cosine/sine)
        if self.filter == 'cosine':
            sinusoidal = torch.cos(self.omega * x_theta) * torch.exp(self.k ** 2 / 2)
        elif self.filter == 'sine':
            sinusoidal = torch.sin(self.omega * x_theta) * torch.exp(self.k ** 2 / 2)
        else:
            raise Exception(f'Unknown Gabor filter type {self.filter}. Should be "cosine" or "sine".')
        
        kernel = gaussian * sinusoidal
        return kernel / kernel.sum()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Apply Gabor filter on a 3D volume.
        """
        
        # Create Gabor kernel
        kernel = self.create_gabor_kernel().unsqueeze(0).unsqueeze(0)
        kernel = kernel.repeat((input.size(1), 1, 1, 1, 1))

        # Apply Gabor filter
        output = F.conv3d(input, kernel, padding='same', groups=input.size(1))

        return output