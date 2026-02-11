import torch
import numpy as np
import os
import nibabel as nib
import argparse
import random

from datasets import MultimodalNIFTIDataset3D
from spatial_diffusion_model import AffineTransformer
from spatial_diffusion import SpatialDiffusion

if __name__ == '__main__':

    # Example command line call:
    # >>> python3 inference_spatial_diffusion.py --ref ../path/to/scan.nii.gz --mov ../path/to/moving/scans/*.nii.gz --ckpt ./path/to/model/ckpts/spatial_diffusion_ema_ckpt_XXXXXX.pth --dataset octa --gpu 0

    parser = argparse.ArgumentParser()

    # General settings
    parser.add_argument('--ref', type=str, required=True, help='Path to reference volume.')
    parser.add_argument('--mov', nargs='+', type=str, required=True, help='Path to a batch of moving volume(s).')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to the spatial diffusion model checkpoint.')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset imaging modality and preprocessing type (\'oct\', \'octa\', \'mri\', or \'venous\').')
    parser.add_argument('--workers', type=int, default=8, help='Number of dataloader worker threads.')
    parser.add_argument('--gpu', type=int, default=None, help='Inference model on a GPU device.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for inferencing.')
    parser.add_argument('--save_path', type=str, default='./results/inference/', help='Folder for storing inference output.')

    # Inferencing settings (keep consistent with training hyperparameters)
    parser.add_argument('--vol_size', type=int, nargs=3, default=[245, 384, 245], help='Cropped volume dimensions.')
    parser.add_argument('--diffusion_steps', type=int, default=65, help='Number of spatial diffusion steps')
    parser.add_argument('--n_classes', type=int, default=1, help='Number of modality-conditional classes for the spatial diffusion model.')
    parser.add_argument('--cond_class', type=int, default=None, help='Modality-conditional class for the spatial diffusion model.')
    parser.add_argument('--bidir', action='store_true', help='Apply bidirectional registration where the reference and transforming volumes are switched.')

    args = parser.parse_args()
    print()

    # Wrap multi-argument list of moving volume files and input volume size
    args.mov = tuple(args.mov)
    args.vol_size = tuple(args.vol_size)

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Extract reference and moving volume metadata
    meta = []
    for i in [args.ref] + list(args.mov):
        vol = nib.load(args.ref)
        meta.append((vol.affine, vol.shape, vol.header.get_data_dtype()))

    # Set inferencing device
    if args.gpu is None:
        # No GPU specified -- set device to CPU
        device = torch.device('cpu')
        print('Using CPU...')
    else:
        # Check if GPU is available
        if not torch.cuda.is_available():
            raise Exception('Tried to find GPU, but no devices are available!')
        
        # Use GPU device for inferencing
        gpu_device = f'cuda:{args.gpu}'
        device = torch.device(gpu_device)
        print(f'Using GPU ({gpu_device})...')

    # Determine final kernel size
    depth, height, width = args.vol_size
    for _ in range(6):
        depth = (depth - 1) // 2 + 1
        height = (height - 1) // 2 + 1
        width = (width - 1) // 2 + 1

    output_dims = (depth, height, width)

    # Check that a valid number of classes are specified
    if args.n_classes <= 0:
        raise ValueError(f'Number of classes must be a positive integer, but got {args.n_classes} instead!')
    
    # Initialize spatial diffusion model
    model = AffineTransformer(input_dim=args.vol_size, n_classes=args.n_classes, output_kernel=output_dims).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    # Initialize diffuser
    diffusion = SpatialDiffusion(args.diffusion_steps, interp='bilinear', pbar_leave=False, device=device)

    # Initialize dataloaders for reference and moving volumes
    dataloaders = {
        'ref': MultimodalNIFTIDataset3D([[args.ref]], args.vol_size, types=[args.dataset], batch_size=1, 
                                        shuffle=False, num_workers=args.workers, persistent_workers=True),
        'mov': MultimodalNIFTIDataset3D([args.mov], args.vol_size, types=[args.dataset], batch_size=len(args.mov), 
                                        shuffle=False, num_workers=args.workers, persistent_workers=True)
    }

    # Load reference and moving volumes
    ref, v_range_ref, _, _ = iter(dataloaders['ref']).__next__()
    mov, v_range_mov, _, _ = iter(dataloaders['mov']).__next__()

    # Move reference and moving volumes to device
    ref = ref.to(device)
    mov = mov.to(device)

    # Stack reference and moving volumes into a batch
    input = []
    for i in range(mov.size(0)):
        input.append(torch.cat([ref[0], mov[i]], dim=0))
    input = torch.stack(input, dim=0)

    # Define modality-conditional class
    if args.cond_class is None:
        class_cond = None
    else:
        class_cond = args.cond_class * torch.ones(input.size(0), device=device, dtype=torch.int32)

    # Perform registration
    if args.bidir:
        reg, transf = diffusion.predict_bidirectional(
            input, model, prepr_type=args.dataset, class_cond=class_cond
        )[0]
    else:
        reg, transf = diffusion.predict(
            input, model, prepr_type=args.dataset, class_cond=class_cond
        )

    # Unnormalize the registered volumes
    reg = reg * 0.5 + 0.5
    reg = reg.detach().cpu().numpy()
    transf = transf.detach().cpu().numpy()

    # Get the intensity normalization ranges
    v_min_ref, v_max_ref = v_range_ref[:,0].item(), v_range_ref[:,1].item()
    v_min_mov, v_max_mov = v_range_mov[:,0].detach().cpu().numpy(), v_range_mov[:,1].detach().cpu().numpy()
    reg = reg * (v_max_mov.reshape(-1, 1, 1, 1, 1) - v_min_mov.reshape(-1, 1, 1, 1, 1))
    reg += v_min_mov.reshape(-1, 1, 1, 1, 1)

    # Gather all input volumes
    vols = [ref[0,0].detach().cpu().numpy()] + [mov[i,0].detach().cpu().numpy() for i in range(mov.size(0))]

    # Create output directory if necessary
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    # Save preprocessed input and registered output volumes
    print('Writing input/output files...')
    for i, (v, (aff, raw_shape, dtype)) in enumerate(zip(vols, meta)):
        # The first volume is the reference, subsequent ones are the moving volumes
        save_path = f'{args.save_path}/ref.nii.gz' if i == 0 else f'{args.save_path}/mov_{i}.nii.gz'

        # Create scaling matrix based on the ratio of dimensions (identity if volume was not resized)
        S = np.diag([
            raw_shape[0] / v.shape[0],
            raw_shape[1] / v.shape[1],
            raw_shape[2] / v.shape[2], 1
        ])

        # Normalize the volumes between 0 - 1
        v = v * 0.5 + 0.5

        # Unnormalize the input volume and write the registered volumes
        if i == 0:
            v = v * (v_max_ref - v_min_ref) + v_min_ref
        else:
            v = v * (v_max_mov[i-1] - v_min_mov[i-1]) + v_min_mov[i-1]

            # Scale the affine matrix and write registered volumes
            nifti = nib.Nifti1Image(reg[i-1,0].astype(dtype), aff @ S)
            nib.save(nifti, f'{args.save_path}/registered_{i}.nii.gz')

            # Save the registration transformation matrix
            np.save(f'{args.save_path}/transf_matrix_{i}.npy', transf[i-1])

        # Write the input volumes
        nifti = nib.Nifti1Image(v.astype(dtype), aff)
        nib.save(nifti, save_path)
    print('Done.\n')