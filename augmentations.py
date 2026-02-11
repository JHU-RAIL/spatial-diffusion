import math
import random
import torch
import torch.nn.functional as F
from typing import Tuple, Union, Optional

class StretchArtifact3d(torch.nn.Module):
    def __init__(
        self,
        stretch_size: Union[int, Tuple[int]] = (7, 14),
        num_artifacts: Union[int, Tuple[int]] = (3, 6),
        direction: str = 'horizontal'
    ) -> None:
        
        super().__init__()

        """
        Applies a stretching artifact to the input volume. 
        Expects volumetric Tensor input of B x C x D x H x W.
        """

        if not isinstance(stretch_size, Tuple):
            # Set stretch size to (1, |stretch_size|)
            self.stretch_size = (1, abs(stretch_size))
        else:
            if stretch_size[0] < 1 or stretch_size[1] < 1:
                raise Exception('Stretch size range must be positive')

            self.stretch_size = stretch_size


        if not isinstance(num_artifacts, Tuple):
            # Set number of artifacts to a constant value
            self.num_artifacts = (abs(num_artifacts), abs(num_artifacts))
        else:
            if num_artifacts[0] < 0 or num_artifacts[1] < 0:
                raise Exception('Number of artifacts range must be non-negative')

            self.num_artifacts = num_artifacts


        # Check that direction is either 'horizontal', 'vertical', or 'both'
        direction = direction.lower()
        dir_choices = ['horizontal', 'vertical', 'both', 'either']

        if direction not in dir_choices:
            raise Exception('Artifact direction should be either \'horizontal\', ' \
                            '\'vertical\', \'both\', or \'either\'.')
        else:
            self.direction = direction

    def forward(
        self,
        input: torch.Tensor,
        i_x: Optional[torch.Tensor] = None,
        i_y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        # Apply stretch artifacts in the specified direction
        if self.direction == 'horizontal':
            return self._horizontal(input, i_x)
        
        elif self.direction == 'vertical':
            return self._vertical(input, i_y)
        
        elif self.direction == 'both':
            return self._vertical(self._horizontal(input, i_x), i_y)

        elif self.direction == 'either':
            return self._horizontal(input, i_x) if random.random() >= 0.5 else self._vertical(input, i_y)
        
        else:
            raise Exception(f'Cannot apply artifact in unknown direction "{self.direction}"')

    def _horizontal(
        self,
        input: torch.Tensor,
        i_x: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        _, C, D, _, W = input.size()

        num_artifacts = torch.tensor(self.num_artifacts[0], device=input.device)
        if self.num_artifacts[0] != self.num_artifacts[1]:
            # Randomly sample number of artifacts to insert
            num_artifacts = torch.randint(self.num_artifacts[0], self.num_artifacts[1], (1,), device=input.device)

        stretched = input.clone()

        # Randomly generate sequence of possible indices to insert artifacts
        if i_x is not None:
            pos = i_x[:num_artifacts]
        else:
            pos = torch.randperm(input.size(3), device=input.device)[:num_artifacts]

        # Iterate through the number of artifacts to add
        for i in range(num_artifacts):
            stretch = self.stretch_size[0]
            if self.stretch_size[0] != self.stretch_size[1]:
                # Randomly select size of the stretch artifact
                stretch = torch.randint(self.stretch_size[0], self.stretch_size[1], (1,), device=input.device)

            # Sample position of the artifact & ensure it stays within dimensions of image
            p = pos[i] if pos[i] + stretch < input.size(3) else input.size(3) - stretch

            # Apply the stretch artifact by copying pixels from a single horizontal frame
            stretched[:,:,:,p:p+stretch] = input[:,:,:,p].view(-1, C, D, 1, W)
        
        return stretched
    
    def _vertical(
        self,
        input: torch.Tensor,
        i_y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        _, C, D, H, _ = input.size()

        num_artifacts = torch.tensor(self.num_artifacts[0], device=input.device)
        if self.num_artifacts[0] != self.num_artifacts[1]:
            # Randomly sample number of artifacts to insert
            num_artifacts = torch.randint(self.num_artifacts[0], self.num_artifacts[1], (1,), device=input.device)

        stretched = input.clone()

        # Randomly generate sequence of possible indices to insert artifacts
        if i_y is not None:
            pos = i_y[:num_artifacts]
        else:
            pos = torch.randperm(input.size(4), device=input.device)[:num_artifacts]

        # Iterate through the number of artifacts to add
        for i in range(num_artifacts):
            stretch = self.stretch_size[0]
            if self.stretch_size[0] != self.stretch_size[1]:
                # Randomly select size of the stretch artifact
                stretch = torch.randint(self.stretch_size[0], self.stretch_size[1], (1,), device=input.device)

            # Sample position of the artifact & ensure it stays within dimensions of image
            p = pos[i] if pos[i] + stretch < input.size(4) else input.size(4) - stretch

            # Apply the stretch artifact by copying pixels from a single vertical frame
            stretched[:,:,:,:,p:p+stretch] = input[:,:,:,:,p].view(-1, C, D, H, 1)
        
        return stretched

class ShadowArtifact3d(torch.nn.Module):
    def __init__(
        self,
        control_point_grid: Union[int, Tuple[int, int, int]] = (12, 6, 6)
    ) -> None:

        super().__init__()

        """
        Applies a shadow artifact to the input volume. 
        Expects volumetric Tensor input of B x C x D x H x W.
        """

        if not isinstance(control_point_grid, Tuple):
            # Set control point grid dimensions to (control_point_grid, 
            # control_point_grid, control_point_grid)
            self.control_point_grid = (
                control_point_grid, control_point_grid, control_point_grid
            )
        else:
            self.control_point_grid = control_point_grid
    
    def forward(
        self,
        input: torch.Tensor,
    ) -> None:

        B, C, D, H, W = input.size()

        # Determine if input is normalized between -1 - 1 or in range 0 - 1
        zero_centered_norm = True if input.min() < 0 else False

        # Determine if input is un-normalized in range 0 - 255
        normalized = zero_centered_norm or input.max() <= 1

        # Un-normalize if input is noramlized between -1 - 1
        if zero_centered_norm:
            input = input * 0.5 + 0.5

        # Create a grid of control points with weights random weights
        # sampled from a uniform distribution
        grid = torch.rand((B, C, *self.control_point_grid), device=input.device)

        # Interpolate grid up to original image dimensions and clamp between 0 - 1
        mask = torch.nn.functional.interpolate(
            grid, size=(D, H, W), mode='trilinear', align_corners=False
        )

        # Apply mask on the input and re-normalize between 0 - 1
        output = input * mask
        output = (output - output.min()) / (output.max() - output.min())

        # Un-normalize input to 0 - 255 range if necessary
        if not normalized:
            output *= 255

        # Re-normalize input between -1 - 1 if necessary
        if zero_centered_norm:
            output = (output - 0.5) / 0.5

        return output

class MotionArtifact3d(torch.nn.Module):
    def __init__(
        self,
        motion_size: Union[int, Tuple[int, int]] = (3, 15),
        translation: Union[int, Tuple[int, int]] = (6, 30),
        num_artifacts: Union[int, Tuple[int, int]] = (3, 13),
        direction: str = 'horizontal'
    ) -> None:
        
        super().__init__()

        """
        Applies a motion artifact to the input volume. 
        Expects volumetric Tensor input of B x C x D x H x W.
        """

        if not isinstance(motion_size, Tuple):
            # Set motion artifact segment size to (1, |stretch_size|)
            self.motion_size = (1, abs(motion_size))
        else:
            if motion_size[0] < 1 or motion_size[1] < 1:
                raise Exception('Motion artifact size range must be positive')

            self.motion_size = motion_size


        if not isinstance(translation, Tuple):
            # Set translation range to (-|translation|, |translation|)
            self.translation = (-abs(translation), abs(translation))
        else:
            self.translation = translation


        if not isinstance(num_artifacts, Tuple):
            # Set number of artifacts to a constant value
            self.num_artifacts = (abs(num_artifacts), abs(num_artifacts))
        else:
            if num_artifacts[0] < 0 or num_artifacts[1] < 0:
                raise Exception('Number of artifacts range must be non-negative')

            self.num_artifacts = num_artifacts


        # Check that direction is either 'horizontal', 'vertical', or 'both'
        direction = direction.lower()
        dir_choices = ['horizontal', 'vertical', 'both', 'either']

        if direction not in dir_choices:
            raise Exception('Artifact direction should be either \'horizontal\', ' \
                            '\'vertical\', \'both\', or \'either\'.')
        else:
            self.direction = direction

    def forward(
        self,
        input: torch.Tensor,
        i_x: Union[torch.Tensor, None] = None,
        i_y: Union[torch.Tensor, None] = None
    ) -> torch.Tensor:

        # Apply stretch artifacts in the specified direction
        if self.direction == 'horizontal':
            return self._horizontal(input, i_x)
        
        elif self.direction == 'vertical':
            return self._vertical(input, i_y)
        
        elif self.direction == 'both':
            return self._vertical(self._horizontal(input, i_x), i_y)
        
        elif self.direction == 'either':
            return self._horizontal(input, i_x) if random.random() > 0.5 else self._vertical(input, i_y)
        
        else:
            raise Exception(f'Cannot apply artifact in unknown direction "{self.direction}"')

    def _horizontal(
        self,
        input: torch.Tensor,
        i_x: Union[torch.Tensor, None] = None
    ) -> torch.Tensor:

        num_artifacts = torch.tensor(self.num_artifacts[0], device=input.device)
        if self.num_artifacts[0] != self.num_artifacts[1]:
            # Randomly sample number of artifacts to insert
            num_artifacts = torch.randint(self.num_artifacts[0], self.num_artifacts[1], (1,), device=input.device)

        motion_vol = input.clone()

        # Randomly generate sequence of possible indices to insert artifacts
        if i_x is not None:
            pos = i_x[:num_artifacts]
        else:
            pos = torch.randperm(input.size(3), device=input.device)[:num_artifacts]

        # Iterate through the number of artifacts to add
        for i in range(num_artifacts):

            translation = torch.tensor(self.translation[0], device=input.device)
            if self.translation[0] != self.translation[1]:
                # Randomly sample translation from uniform distribution
                translation = torch.randint(self.translation[0], self.translation[1], (1,), device=input.device)

            motion = self.motion_size[0]
            if self.motion_size[0] != self.motion_size[1]:
                # Randomly select size of the motion artifact segment
                motion = torch.randint(self.motion_size[0], self.motion_size[1], (1,), device=input.device)

            # Sample position of the artifact & ensure it stays within dimensions of image
            p = pos[i] if pos[i] + motion < input.size(3) else input.size(3) - motion

            # Apply the horizontal motion artifact by translating frames along the x-axis
            motion_vol[:,:,:,p:p+motion] = torch.roll(input[:,:,:,p:p+motion], shifts=(translation,), dims=(4,))
        
        return motion_vol
    
    def _vertical(
        self,
        input: torch.Tensor,
        i_y: Union[torch.Tensor, None] = None
    ) -> torch.Tensor:

        num_artifacts = torch.tensor(self.num_artifacts[0], device=input.device)
        if self.num_artifacts[0] != self.num_artifacts[1]:
            # Randomly sample number of artifacts to insert
            num_artifacts = torch.randint(self.num_artifacts[0], self.num_artifacts[1], (1,), device=input.device)

        motion_vol = input.clone()

        # Randomly generate sequence of possible indices to insert artifacts
        if i_y is not None:
            pos = i_y[:num_artifacts]
        else:
            pos = torch.randperm(input.size(4), device=input.device)[:num_artifacts]

        # Iterate through the number of artifacts to add
        for i in range(num_artifacts):

            translation = torch.tensor(self.translation[0], device=input.device)
            if self.translation[0] != self.translation[1]:
                # Randomly sample translation from uniform distribution
                translation = torch.randint(self.translation[0], self.translation[1], (1,), device=input.device)

            motion = self.motion_size[0]
            if self.motion_size[0] != self.motion_size[1]:
                # Randomly select size of the motion artifact segment
                motion = torch.randint(self.motion_size[0], self.motion_size[1], (1,), device=input.device)

            # Sample position of the artifact & ensure it stays within dimensions of image
            p = pos[i] if pos[i] + motion < input.size(4) else input.size(4) - motion

            # Apply the vertical motion artifact by translating frames along the y-axis
            motion_vol[:,:,:,:,p:p+motion] = torch.roll(input[:,:,:,:,p:p+motion], shifts=(translation,), dims=(3,))
        
        return motion_vol

class DoublingArtifact3d(torch.nn.Module):
    def __init__(
        self,
        translation: Union[int, Tuple[int, int]] = 15,
        alpha_translated: Union[float, Tuple[float, float]] = 0.35,
        xyz_dims: Tuple[int, int, int] = (-1, -2, -3)
    ) -> None:
        
        super().__init__()

        """
        Applies a doubling artifact to the input volume. 
        Expects volumetric Tensor input of * x D x H x W (* is any number of dimensions).
        """

        if not isinstance(translation, Tuple):
            # Set translation range to (-|translation|, |translation|)
            self.translation = (-abs(translation), abs(translation))
        else:
            self.translation = translation

        if not isinstance(alpha_translated, Tuple):
            # Set translation range to (0, |alpha_translated|)
            self.alpha = (0, abs(alpha_translated))
        else:
            if alpha_translated[0] < 0 or alpha_translated[1] < 0:
                raise Exception('Translated alpha range must be non-negative')
            
            self.alpha = alpha_translated

        # x-axis maps to --> xyz_dims[0]
        # y-axis maps to --> xyz_dims[1]
        # z-axis maps to --> xyz_dims[2]
        self.xyz_dims = xyz_dims

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        # Randomly generate alpha for the input
        alpha = self.alpha[0] + (self.alpha[1] - self.alpha[0]) * torch.rand(1, device=input.device)

        translation_x = torch.tensor(self.translation[0], device=input.device)
        translation_z = torch.tensor(self.translation[0], device=input.device)
        
        if self.translation[0] != self.translation[1]:
            # Randomly sample translations in the x-axis and z-axis from uniform distribution
            translation_x = torch.randint(self.translation[0], self.translation[1], (1,), device=input.device)
            translation_z = torch.randint(self.translation[0], self.translation[1], (1,), device=input.device)

        # Apply translation on the input
        translated = torch.roll(input, shifts=(translation_x.item(), translation_z.item()), 
                                dims=(self.xyz_dims[0], self.xyz_dims[1]))

        # Create doubling artifact by taking a weighted average of the input and translated
        doubled = (1 - alpha) * input + alpha * translated
        
        return doubled

class RandomDeformation(torch.nn.Module):
    def __init__(
        self,
        input_range: Tuple[float, float] = (-1.0, 1.0),
        control_point_grid: Union[int, Tuple[int, int, int]] = (12, 6, 6),
        std_displ: Union[float, Tuple[float, float, float]] = 3.0
    ) -> None:
        super().__init__()

        # Input range should be a tuple containing (min, max) intensities
        if not isinstance(input_range, Tuple) or len(input_range) != 2:
            raise ValueError('Input range must be a tuple containing (min, max) intensities!')

        if not isinstance(control_point_grid, Tuple):
            # Set control point grid dimensions to (control_point_grid, 
            # control_point_grid, control_point_grid)
            self.control_point_grid = (
                control_point_grid, control_point_grid, control_point_grid
            )
        else:
            self.control_point_grid = control_point_grid

        if not isinstance(std_displ, Tuple):
            # Convert standard deviation of deformation into a 3D tuple
            self.std_displ = (std_displ, std_displ, std_displ)
        else:
            self.std_displ = std_displ

        self.input_range = input_range

    def forward(
        self,
        input: torch.Tensor,
        trunc_sigma: Union[float, None] = 2.0
    ) -> torch.Tensor:

        B, C, D, H, W = input.size()

        # Randomly sample a displacement field from a normal distribution
        displ_field = torch.randn((B, 3, *self.control_point_grid), device=input.device)
        
        # Truncate values if necessary
        if trunc_sigma is not None:
            displ_field = torch.clamp(displ_field, -trunc_sigma, trunc_sigma)

        # Displacement field based on the specified standard deviations for the xyz-axis
        # Scaling by 2 is due to PyTorch's displacement field range being between -1 - 1
        displ_field[:,0] *= self.std_displ[0] * 2.0 / D
        displ_field[:,1] *= self.std_displ[1] * 2.0 / H
        displ_field[:,2] *= self.std_displ[2] * 2.0 / W

        # Interpolate displacements and compute deformation from the displacement field
        displ_field = F.interpolate(displ_field, size=(D, H, W), mode='trilinear', align_corners=False)
        deformation = displ_field.permute(0, 2, 3, 4, 1) + self._get_identity_grid(input)

        # Normalize input between 0 - 1
        input = (input - self.input_range[0]) / (self.input_range[1] - self.input_range[0])

        # Apply deformation
        deformed = F.grid_sample(input, deformation, mode='bilinear', align_corners=False)

        # Unnormalize transformed output back to original input intensity range
        deformed = deformed * (self.input_range[1] - self.input_range[0]) + self.input_range[0]

        return deformed

    def _get_identity_grid(self, input: torch.Tensor) -> torch.Tensor:
        """
        Create a 3D volume identity mapping grid normalized between [-1, 1].
        """
        B, _, D, H, W = input.size()

        # Define grid across each of the three axes (depth, height, width)
        d = torch.linspace(-1, 1, steps=D, device=input.device)
        h = torch.linspace(-1, 1, steps=H, device=input.device)
        w = torch.linspace(-1, 1, steps=W, device=input.device)
        grid_d, grid_h, grid_w = torch.meshgrid(d, h, w, indexing='ij')

        # Concatenate grids for each of the three dimensions
        identity = torch.stack([grid_w, grid_h, grid_d], dim=-1)
        identity = identity.unsqueeze(0).repeat(B, 1, 1, 1, 1)

        return identity


class RandomCrop(torch.nn.Module):
    def __init__(
        self,
        crop_size: Tuple[int, int] = (190, 245),
        pad_value: int = 0
    ) -> None:
        super().__init__()

        self.crop_size = crop_size
        self.pad_value = pad_value
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:

        B, _, D, _, W = input.size()

        # Initialize the output tensor with the specified pad value
        out = torch.full_like(input, self.pad_value)

        for i in range(B):
            # Randomize the crop size within the specified range
            crop_d = random.randint(self.crop_size[0], self.crop_size[1])
            crop_w = random.randint(self.crop_size[0], self.crop_size[1])

            # Randomly select the top-left corner of the crop
            top = random.randint(0, D - crop_d)
            left = random.randint(0, W - crop_w)

            # Crop the region from the input tensor with respect to the x-z plane
            crop = input[i,:,top:top + crop_d,:,left:left + crop_w]

            # Place the cropped region into the output tensor
            out[i,:,top:top + crop_d,:,left:left + crop_w] = crop

        return out

class RandomRigid(torch.nn.Module):
    def __init__(
        self,
        input_range: Tuple[float, float] = (-1.0, 1.0), 
        angle_xy: Tuple[float, float] = (0.0, 0.0),
        angle_xz: Tuple[float, float] = (-180.0, 180.0),
        angle_yz: Tuple[float, float] = (0.0, 0.0),
        transl: Tuple[float, float] = (-30.0, 30.0)
    ) -> None:
        super().__init__()

        # Input range should be a tuple containing (min, max) intensities
        if not isinstance(input_range, Tuple) or len(input_range) != 2:
            raise ValueError('Input range must be a tuple containing (min, max) intensities!')

        self.input_range = input_range
        self.angle_xy = angle_xy
        self.angle_xz = angle_xz
        self.angle_yz = angle_yz
        self.transl = transl
    
    def forward(self, input: torch.Tensor, return_mat: bool = False) -> torch.Tensor:

        B = input.size(0)

        I = torch.tensor([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0]
            ], device=input.device)[None,].repeat(B, 1, 1)

        angles_xy = (self.angle_xy[0] + (self.angle_xy[1] - self.angle_xy[0]) * torch.rand(B, device=input.device)) * (math.pi / 180.0)
        angles_xz = (self.angle_xz[0] + (self.angle_xz[1] - self.angle_xz[0]) * torch.rand(B, device=input.device)) * (math.pi / 180.0)
        angles_yz = (self.angle_yz[0] + (self.angle_yz[1] - self.angle_yz[0]) * torch.rand(B, device=input.device)) * (math.pi / 180.0)
        translations = (self.transl[0] + (self.transl[1] - self.transl[0]) * torch.rand((3, B), device=input.device))
        
        # Compute transformation matrices of the specified angles and translations
        transf_mat = torch.stack([self._get_transf_matrix((angles_yz[i], angles_xz[i], angles_xy[i]), 
                                  (translations[0,i], translations[1,i], translations[2,i]), 
                                  input.shape[2:], input.device) for i in range(B)])
        grid = F.affine_grid(transf_mat, input.size(), align_corners=False)

        # Normalize input between 0 - 1
        input = (input - self.input_range[0]) / (self.input_range[1] - self.input_range[0])

        # Apply transformation on the input
        transformed = F.grid_sample(input, grid, mode='bilinear', align_corners=False)

        # Unnormalize transformed output back to original input intensity range
        transformed = transformed * (self.input_range[1] - self.input_range[0]) + self.input_range[0]

        if return_mat:
            # Compute inverse transformation matrices of the specified angles and translations
            rev_transf_mat = torch.stack([self._get_transf_matrix((-angles_yz[i], -angles_xz[i], -angles_xy[i]), 
                                    (-translations[0,i], -translations[1,i], -translations[2,i]), 
                                    input.shape[2:], input.device) for i in range(B)])

            return transformed, transf_mat, rev_transf_mat
        else:

            return transformed
    
    def _get_transf_matrix(
        self,
        angle: Union[Tuple[torch.Tensor, ...], torch.Tensor],
        translation: Union[Tuple[torch.Tensor, ...], torch.Tensor],
        dims: Tuple[int, int, int],
        device: torch.device = None
    ) -> torch.Tensor:
        """
        Generate a 3D affine transformation matrix for rotation around the x, y, and z-axis
        translations in the x, y, and z-axis.
        """

        # If a scalar angle is specified, default to using same angle in each axis
        if not isinstance(angle, Tuple):
            angle = (angle, angle, angle)

        # If a scalar translation is specified, default to using same translation in each axis
        if not isinstance(translation, Tuple):
            translation = (translation, translation, translation)

        # Rotation matrix around the x-axis
        R_x = torch.tensor([[1.0, 0.0, 0.0],
                            [0.0, math.cos(angle[0]), -math.sin(angle[0])],
                            [0.0, math.sin(angle[0]), math.cos(angle[0])]], device=device)
    
        # Rotation matrix around the y-axis
        R_y = torch.tensor([[math.cos(angle[1]), 0.0, math.sin(angle[1])],
                            [0.0, 1.0, 0.0],
                            [-math.sin(angle[1]), 0.0, math.cos(angle[1])]], device=device)
        
        # Rotation matrix around the z-axis
        R_z = torch.tensor([[math.cos(angle[2]), -math.sin(angle[2]), 0],
                            [math.sin(angle[2]), math.cos(angle[2]), 0],
                            [0.0, 0.0, 1.0]], device=device)

        # Translation vector in the x, y, and z-axis
        transl = torch.tensor([
            [translation[0] / dims[0]],
            [translation[1] / dims[1]],
            [translation[2] / dims[2]]
        ], device=device)

        # Combine transformation and translation components to create a 
        # rigid transformation matrix
        transf_matrix = torch.cat([R_x @ R_y @ R_z, transl], dim=1)
        
        return transf_matrix
