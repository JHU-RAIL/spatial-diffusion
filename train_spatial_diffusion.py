import torch
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import random
import math

from datasets import MultimodalNIFTIDataset3D, DatasetLoader
from augmentations import RandomCrop, RandomRigid, RandomDeformation, MotionArtifact3d, DoublingArtifact3d, ShadowArtifact3d
from spatial_diffusion_model import AffineTransformer
from spatial_diffusion import SpatialDiffusion, Transformation
from ema_pytorch import EMA
from fvcore.nn import FlopCountAnalysis, flop_count_table

if __name__ == '__main__':

    # Example command line calls:
    
    # Train model on individual modalities:
    # >>> python3 train_spatial_diffusion.py --train ./path/to/OCT/files/ --dataset oct --save_path ./results/oct --gpu 0 --workers 12 --steps 115000
    # >>> python3 train_spatial_diffusion.py --train ./path/to/OCTA/files/--dataset octa --save_path ./results/octa --gpu 0 --workers 12 --steps 115000
    # >>> python3 train_spatial_diffusion.py --train ./path/to/OASIS-1/training/files/ --val ./path/to/OASIS-1/validation/files/ --dataset mri --vol_size 256 256 256 --save_path ./results/mri --gpu 0 --workers 12 --steps 120000
    # >>> python3 train_spatial_diffusion.py --train ./path/to/head_and_neck_VM/files/ --dataset venous --vol_size 256 256 256 --save_path ./results/venous --gpu 0 --workers 12 --steps 115000
    
    # Train model on multiple modalities:
    # >>> python3 train_spatial_diffusion.py --train ./path/to/OCT/files/ ./path/to/OCTA/files/ --dataset oct octa --save_path ./results/oct_octa --gpu 0 --workers 12 --steps 165000
    # >>> python3 train_spatial_diffusion.py --train ./path/to/OCT/files/ ./path/to/OCTA/files/ ./path/to/OASIS-1/training/files/ ./path/to/head_and_neck_VM/files/ --val - - ./path/to/OASIS-1/validation/files/ - --dataset oct octa mri venous --save_path ./results/foundation_oct-octa-mri-vm --gpu 0 --workers 12 --steps 265000

    parser = argparse.ArgumentParser()

    # General settings
    parser.add_argument('--train', type=str, nargs='+', required=True, help='Training dataset folder of nifti files or .pkl file.')
    parser.add_argument('--val', type=str, nargs='+', default=None, help='Validation dataset folder or .pkl file. 90/10 training split if \'-\' or not specified.')
    parser.add_argument('--dataset', type=str, nargs='+', default=None, help='Dataset imaging modality and preprocessing type (\'oct\', \'octa\', \'mri\', or \'venous\').')
    parser.add_argument('--workers', type=int, default=8, help='Number of dataloader worker threads.')
    parser.add_argument('--gpu', type=int, default=None, help='Train model on a GPU device.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for training.')
    parser.add_argument('--uncond', action='store_true', help='Force the model to train unconditionally regardless of the number of datasets.')

    # Training and logging steps
    parser.add_argument('--warmup', type=int, default=5000, help='Number of warmup steps for learning rate scheduler.')
    parser.add_argument('--steps', type=int, default=115000, help='Number of training steps.')
    parser.add_argument('--train_loss_steps', type=int, default=200, help='Interval, in steps, for logging training loss.')
    parser.add_argument('--val_loss_steps', type=int, default=1000, help='Interval, in steps, for computing validation loss.')
    parser.add_argument('--eval_steps', type=int, default=5000, help='Interval, in steps, for inferencing the model.')
    parser.add_argument('--save_path', type=str, default='./results/registration', help='Folder for storing training curve and inference output.')

    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=4, help='Training batch size.')
    parser.add_argument('--grad_accum', type=int, default=1, help='Number of gradient accumulations.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Base learning rate for training.')
    parser.add_argument('--vol_size', type=int, nargs=3, default=[245, 384, 245], help='Input volume dimensions.')
    parser.add_argument('--ema_decay', type=float, default=0.995, help='Exponential moving average decay rate.')
    parser.add_argument('--ema_power', type=float, default=3. / 4, help='Exponential moving average power constant.')
    parser.add_argument('--affine_prob', type=float, default=0.8, help='Probability of training on an affine instead of a rigid-body sample.')
    parser.add_argument('--class_dropout', type=float, default=0.15, help='Modality-conditional class dropout probability to the modality-invarient embedding.')

    # Spatial diffusion hyperparameters
    parser.add_argument('--diffusion_steps', type=int, default=65, help='Number of spatial diffusion steps')
    parser.add_argument('--transf_range', type=float, nargs=2, default=[0.01, 0.07], help='Number of training steps')
    parser.add_argument('--transl_range', type=float, nargs=2, default=[0.5, 4.0], help='Number of training steps')

    args = parser.parse_args()
    print()

    # Wrap multi-argument inputs as a tuple
    args.vol_size = tuple(args.vol_size)
    args.transf_range = tuple(args.transf_range)
    args.transl_range = tuple(args.transl_range)

    # Make all the dataset types case invarient
    args.dataset = [type.lower() for type in args.dataset]

    # Check that the volume size is D x H x W
    if len(args.vol_size) != 3:
        raise ValueError('Expected 3 arguments for the args.vol_size parameter for D, H, W.')

    # Check that the transformation range is min, max
    if len(args.transf_range) != 2:
        raise ValueError('Expected 2 arguments for the transf_range parameter for min, max transformation magnitude.')

    # Check that the translation range is min, max
    if len(args.transl_range) != 2:
        raise ValueError('Expected 2 arguments for the transl_range parameter for min, max translation magnitude.')

    # Define output directories
    CKPT_PATH = f'{args.save_path}/ckpts/'
    OUT_PATH = f'{args.save_path}/out/'

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Maintain a list of training and validation files for each dataset
    train_files = []
    val_files = []

    # Keep track of the total training and validation files
    num_train = 0
    num_val = 0

    # Load training and validation files (or split training set if necessary)
    loader = DatasetLoader()
    for i, train_folder in enumerate(args.train):
        # Obtain train and val splits
        val_folder = None if args.val is None or args.val[i] == '-' else args.val[i]
        train, val = loader.load(train_folder, val_folder)

        # Add splits to the list
        train_files.append(train)
        val_files.append(val)

        # Increment the counters for total training and validation files
        num_train += len(train_files[-1])
        num_val += len(val_files[-1])

        # Write training and validation splits as a file
        name = args.dataset[i].lower()
        loader.write(f'{OUT_PATH}training_set_{name}-{i+1}.pkl', f'{OUT_PATH}validation_set_{name}-{i+1}.pkl')
    
    print(f'\nThere are {num_train} training files.')
    print(f'There are {num_val} validation files.')

    # Create checkpoint directory if necessary
    if not os.path.exists(os.path.normpath(CKPT_PATH)):
        os.makedirs(CKPT_PATH)
    else:
        print('Checkpoint path exists, no file directory created')

    # Initialize dataloaders
    dataloaders = {
        'train': MultimodalNIFTIDataset3D(train_files, args.vol_size, types=args.dataset, batch_size=args.batch_size, 
                                          shuffle=True, num_workers=args.workers, persistent_workers=True),
        'val': MultimodalNIFTIDataset3D(val_files, args.vol_size, types=args.dataset, batch_size=args.batch_size, 
                                        shuffle=True, num_workers=args.workers, persistent_workers=True)
    }

    # Set training device
    if args.gpu is None:
        # No GPU specified -- set device to CPU
        device = torch.device('cpu')
        print('Using CPU...')
    else:
        # Check if GPU is available
        if not torch.cuda.is_available():
            raise Exception('Tried to find GPU, but no devices are available!')
        
        # Use GPU device for training
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
    n_classes = 1 if args.uncond else len(args.dataset)

    # Initialize Affine Transformer Model and Adam optimizer
    model = AffineTransformer(input_dim=args.vol_size, n_classes=n_classes, output_kernel=output_dims, 
                              class_dropout=args.class_dropout).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Learning rate scheduler with linear warm-up and cosine annealing
    lr_func = lambda step: np.minimum(
        (step + 1) / (args.warmup + 1e-8),
        0.5 * (math.cos((step - args.warmup) / (args.steps - args.warmup) * math.pi) + 1)
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=False)

    # Analyze Affine Transformer model architecture
    flops_affine_transf = FlopCountAnalysis(model, (torch.randn((1,2,*args.vol_size), device=device), torch.randn(1, device=device),
                                                    'octa', torch.ones(1, device=device).int() if model.n_classes > 1 else None))
    print(f'Affine Transformer Model Breakdown:\n{flop_count_table(flops_affine_transf, max_depth=5)}')

    train_loss = []
    train_loss_steps = []
    val_loss = []
    val_loss_steps = []

    # Initialize helper classes that apply random cropping and rigid-body transformation augmentations
    crop = RandomCrop(pad_value=-1)
    transf = RandomRigid(angle_xz=(-180.0, 180.0), transl=(-30.0, 30.0))

    # Initialize helper classes that apply synthetic artifacts
    motion_artif = MotionArtifact3d(motion_size=(5, 17), translation=(5, 30), num_artifacts=(0, 10), direction='either')
    doubling_artif = DoublingArtifact3d(translation=25, alpha_translated=0.35)
    shadow_artif = ShadowArtifact3d(control_point_grid=(12, 6, 6))
    deform = RandomDeformation(control_point_grid=(14, 7, 7), std_displ=3.0)

    # Helper function that applies synthetic artifacts to the input volume
    def artifact(input: torch.Tensor, type: str, step: int) -> torch.Tensor:
        """
        Helper function that applies random artifacts on the input.
        """

        # Re-arrange input axes
        output = input.permute(0, 1, 3, 2, 4)

        # Apply deformation
        if type != 'octa' and random.random() < 0.5:
            output = deform(output)

        # Apply artifacts
        if random.random() < 0.5:
            output = motion_artif(output)   # Motion artifact

        if random.random() < 0.5:
            output = doubling_artif(output) # Vessel doubling artifact

        if random.random() < 0.5:
            output = shadow_artif(output)   # Shadow artifact

        # Re-arrange output axes
        output = output.permute(0, 1, 3, 2, 4)

        return output

    # Initialize forward and reverse spatial diffusior helper class (rigid body)
    diffusion_rigid = SpatialDiffusion(args.diffusion_steps, args.transf_range, 
                              args.transl_range, type=Transformation.RIGID_BODY,
                              device=device)

    # Initialize forward and reverse spatial diffusior helper class (affine)
    diffusion_affine = SpatialDiffusion(args.diffusion_steps, args.transf_range, 
                              args.transl_range, type=Transformation.AFFINE,
                              device=device)
    
    # Initialize Exponential Moving Average (EMA) model with the EMA decay and power parameters
    ema = EMA(model, beta=args.ema_decay, update_every=1, inv_gamma=1.0, 
              power=args.ema_power, update_after_step=100)

    # Reset the gradient graph
    optim.zero_grad()

    # Begin the model optimization loop across the total number of training steps
    print('Beginning training...\n')

    # Create iterable train loader to sample from
    train_loader = iter(dataloaders['train'])
    total_steps = args.steps

    # Display progress bar and iterate over model training steps
    for step in (pbar := tqdm(range(total_steps), total=total_steps, desc='Training Progress')):
        # Start training loop - set model to training mode
        model.train()

        losses = []

        # Declare variables storing model input and output for later access when logging
        vol = None
        q_t, vol_fixed, output = None, None, None

        # Accumulate gradients for the specified number of steps
        for _ in range(args.grad_accum):
            # Randomly choose whether to train on a rigid body or an affine transformation
            diffusion = diffusion_affine if random.random() < args.affine_prob else diffusion_rigid

            # Load the next batch of training samples from the dataloader
            try:
                vol, _, type, class_id = next(train_loader)
            except StopIteration:
                train_loader = iter(dataloaders['train'])
                vol, _, type, class_id = next(train_loader)

            # Apply augmentations to the data sample
            vol = vol.to(device)
            vol = transf(vol.to(device))

            # Set crop size based on modality
            offset_factor = 0.25 if type == 'oct' or type == 'octa' else 0.40
            crop.crop_size = (args.vol_size[0] - int(args.vol_size[0] * offset_factor), args.vol_size[0])

            # Define modality-conditional class ID
            class_id = ((args.class_dropout > 0) + class_id).to(device).int() if model.n_classes > 1 else None

            # Sample random timestep(s)
            timesteps = diffusion.sample_timesteps(vol.size(0), device=device)

            # Sample forward diffusion transformation(s) up to sampled timestep(s)
            q_t, gt_rev_affine = diffusion.sample_at_timesteps(
                model.unnormalize(crop(artifact(vol, type, step))), timesteps
            )
            q_t = model.normalize(q_t)

            # Predict the reverse diffusion transformation(s) at the sampled timestep(s)
            vol_fixed = crop(artifact(vol, type, step))
            input = torch.cat([vol_fixed, q_t], dim=1)
            output, affine = model(input, timesteps, type, class_id)
            
            # Compute Frobenius Norm loss between the predicted and ground-truth affine matrix
            loss = torch.linalg.norm(gt_rev_affine - affine, ord='fro', dim=(1, 2)).mean()
            loss /= args.grad_accum
            
            # Backpropogate gradients
            loss.backward()
            losses.append(loss.item())
        
        # Step the optimizer and reset the gradient graph
        optim.step()
        optim.zero_grad()

        # Update the EMA model after every training step
        ema.update()

        # Step the learning rate scheduler after every training step
        lr_scheduler.step()

        # Compute average loss for the current training step
        avg_train_loss = sum(losses)

        # Update progress bar
        pbar.set_postfix({'loss': f'{avg_train_loss:.3}'})

        # Determine if current step is a training loss logging step
        is_train_loss_step = (step + 1) == 1 or (step + 1) % args.train_loss_steps == 0 or \
                             (step + 1) == args.steps

        # Determine if current step is a training loss logging step
        if is_train_loss_step:
            train_loss.append(avg_train_loss)
            train_loss_steps.append(step + 1)

        # Randomly choose whether to evaluate on a rigid body or an affine transformation
        diffusion = diffusion_affine if random.random() < args.affine_prob else diffusion_rigid
        
        # Determine if current step is an eval step
        is_eval_step = (step + 1) == 1 or (step + 1) % args.val_loss_steps == 0 or \
                       (step + 1) == args.steps

        # Evaluate the model
        if is_eval_step:
            # Start validation loop - set model to eval mode
            model.eval()
            ema.ema_model.eval()

            losses = []

            print(f'\nEvaluating model ({num_val} samples)...')
            with torch.inference_mode():
                for vol, _, type, class_id in iter(dataloaders['val']):
                    # Apply augmentations to the data sample
                    vol = transf(vol.to(device))

                    # Set crop size based on modality
                    offset_factor = 0.25 if type == 'oct' or type == 'octa' else 0.40
                    crop.crop_size = (args.vol_size[0] - int(args.vol_size[0] * offset_factor), args.vol_size[0])

                    # Define modality-conditional class ID
                    class_id = ((args.class_dropout > 0) + class_id).to(device).int() if model.n_classes > 1 else None

                    # Sample random timestep(s)
                    timesteps = diffusion.sample_timesteps(vol.size(0), device=device)

                    # Sample forward diffusion transformation(s) up to sampled timestep(s)
                    q_t, gt_rev_affine = diffusion.sample_at_timesteps(
                        model.unnormalize(crop(artifact(vol, type, step))), timesteps
                    )
                    q_t = model.normalize(q_t)

                    # Predict the reverse diffusion transformation(s) at the sampled timestep(s)
                    vol_fixed = crop(artifact(vol, type, step))
                    input = torch.cat([vol_fixed, q_t], dim=1)
                    output, affine = ema.ema_model(input, timesteps, type, class_id)
                    
                    # Compute Frobenius Norm loss between the predicted and ground-truth affine matrix
                    loss = torch.linalg.norm(gt_rev_affine - affine, ord='fro', dim=(1, 2)).mean()
                    losses.append(loss.item())
                
                # Compute average loss across the validation set
                avg_val_loss = sum(losses) / len(losses)
                val_loss.append(avg_val_loss)
                val_loss_steps.append(step + 1)

                print(f'Evaluation Results: Training Loss = {avg_train_loss:.3}, Validation Loss = {avg_val_loss:.3}')
        
        # Log training progress and model output - set EMA model to eval mode
        ema.ema_model.eval()

        with torch.inference_mode():
            # Determine if current step is a logging step
            is_inference_step = (step + 1) == 1 or (step + 1) % args.eval_steps == 0 or \
                                (step + 1) == args.steps

            # Declare variables storing model input and output for later access when logging
            p_0, diffusion_input = None, None

            if is_inference_step:
                print(f'\nPredicting spatial diffusion model ({args.diffusion_steps} steps)...')

                # Set inference random sampling range to 1.2x the maximum transformation and 
                # translation standard deviation
                max_transf = 1.2 * diffusion_affine.transf_std_schedule[-1].item()
                max_transl = 1.2 * diffusion_affine.transl_std_schedule[-1].item()

                # Apply a random transformation on the last validation sample and inference the model
                p_0, diffusion_input, comp_flow = diffusion.random_transf_predict(
                    vol[0].unsqueeze(0), ema.ema_model, type, 
                    class_id[0].unsqueeze(0) if model.n_classes > 1 else None,
                    transf_magnitude=(-max_transf, max_transf), translation=(-max_transl, max_transl)
                )

                # Display the prediction error
                print(f'Spatial diffusion prediction MAE = {F.l1_loss(vol[0].unsqueeze(0), p_0[0].unsqueeze(0)).item():.4}\n')

            elif is_eval_step:
                print()

            # Plot spatial diffusion model loss curve
            plt.figure(figsize=(10, 6))
            plt.plot(train_loss_steps, train_loss, label='Training Loss', linestyle='-', color='blue', linewidth=2)

            # Plot validation loss as well if available
            if len(val_loss) > 0:
                plt.plot(val_loss_steps, val_loss, label='Validation Loss', linestyle='-', color='red', linewidth=2)

            plt.xlabel('Steps', fontsize=17, labelpad=6)
            plt.ylabel('Loss', fontsize=17, labelpad=8)
            plt.title('Spatial Diffusion Model Loss', fontsize=22, pad=20, fontweight='bold')
            plt.legend(fontsize=15)
            plt.grid(True, linestyle='--', linewidth=1.0, color='gray')
            plt.gca().set_facecolor('#f2f2f2')
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.tight_layout()

            if is_inference_step:
                plt.savefig(f'{OUT_PATH}loss_{step + 1}.png', dpi=300)
            
            if is_train_loss_step or is_inference_step:
                plt.savefig(f'{OUT_PATH}loss.png', dpi=300)

            plt.close()

            if is_inference_step:
                """
                Display the ground-truth original volume, randomly transformed volume,
                model-predicted volume, and pixel-wise error between predicted and 
                ground truth for the full reverse diffusion process prediction.
                """

                mip_truth = torch.mean(vol, dim=3)
                mip_truth = mip_truth.detach().cpu().numpy()

                mip_input = torch.mean(diffusion_input, dim=3)
                mip_input = mip_input.detach().cpu().numpy()

                mip_predicted = torch.mean(p_0, dim=3)
                mip_predicted = mip_predicted.detach().cpu().numpy()

                diff = np.abs(vol.data.detach().cpu().numpy()[0,0,args.vol_size[0] // 2,:,:] - p_0.data.detach().cpu().numpy()[0,0,args.vol_size[0] // 2,:,:])
                diff_mip = np.abs(mip_truth - mip_predicted)

                # Use the pixel intensity range of the ground-truth's mean intensity projection
                # as the reference colorbar range for all other volumes in this sample
                min, max = np.min(mip_truth[0,0,:,:]), np.max(mip_truth[0,0,:,:])

                plt.figure(figsize=[24,12])
                ax = plt.subplot(2,4,1)
                ax.set_title(f'Reference Volume (B-scan)', fontsize=24, pad=22)
                plt.imshow(vol.data.detach().cpu().numpy()[0,0,args.vol_size[0] // 2,:,:], cmap = 'gray', clim=(-1, 1))

                ax = plt.subplot(2,4,2)
                ax.set_title(f'Transforming Volume (B-scan)', fontsize=24, pad=22)
                plt.imshow(diffusion_input.data.detach().cpu().numpy()[0,0,args.vol_size[0] // 2,:,:], cmap = 'gray', clim=(-1, 1))

                ax = plt.subplot(2,4,3)
                ax.set_title(f'Registered Volume (B-scan)', fontsize=24, pad=22)
                plt.imshow(p_0.data.detach().cpu().numpy()[0,0,args.vol_size[0] // 2,:,:], cmap = 'gray', clim=(-1, 1))

                ax = plt.subplot(2,4,4)
                ax.set_title(f'Difference (B-scan)', fontsize=24, pad=22)
                plt.imshow(diff, cmap = 'turbo', clim=(0, 2))
                plt.colorbar()

                ax = plt.subplot(2,4,5)
                ax.set_title(f'Reference Volume (en face)', fontsize=24, pad=22)
                plt.imshow(mip_truth[0,0,:,:], cmap = 'gray', clim=(min, max))

                ax = plt.subplot(2,4,6)
                ax.set_title(f'Transforming Volume (en face)', fontsize=24, pad=22)
                plt.imshow(mip_input[0,0,:,:], cmap = 'gray', clim=(min, max))

                ax = plt.subplot(2,4,7)
                ax.set_title(f'Registered Volume (en face)', fontsize=24, pad=22)
                plt.imshow(mip_predicted[0,0,:,:], cmap = 'gray', clim=(min, max))

                ax = plt.subplot(2,4,8)
                ax.set_title(f'Difference (en face)', fontsize=24, pad=22)
                plt.imshow(diff_mip[0,0,:,:], cmap = 'turbo', clim=(0, max - min))
                plt.colorbar()

                plt.tight_layout(pad=4)
                plt.savefig(f'{OUT_PATH}diffusion_pred_{step + 1}.png')
                plt.close()


                """
                Display the ground-truth original volume, transformed volume at a 
                randomly sampled timestep, and model-predicted reverse diffusion step 
                at that particular timestep.
                """
                
                mip_truth = torch.mean(vol_fixed, dim=3)
                mip_truth = mip_truth.detach().cpu().numpy()

                mip_input = torch.mean(q_t, dim=3)
                mip_input = mip_input.detach().cpu().numpy()

                mip_predicted = torch.mean(output, dim=3)
                mip_predicted = mip_predicted.detach().cpu().numpy()

                diff = np.abs(vol_fixed.data.detach().cpu().numpy()[0,0,args.vol_size[0] // 2,:,:] - output.data.detach().cpu().numpy()[0,0,args.vol_size[0] // 2,:,:])
                diff_mip = np.abs(mip_truth - mip_predicted)

                # Use the pixel intensity range of the ground-truth's mean intensity projection
                # as the reference colorbar range for all other volumes in this sample
                min, max = np.min(mip_truth[0,0,:,:]), np.max(mip_truth[0,0,:,:])

                plt.figure(figsize=[24,12])
                ax = plt.subplot(2,4,1)
                ax.set_title(f'Reference Volume (B-scan)', fontsize=24, pad=22)
                plt.imshow(vol_fixed.data.detach().cpu().numpy()[0,0,args.vol_size[0] // 2,:,:], cmap = 'gray', clim=(-1, 1))

                ax = plt.subplot(2,4,2)
                ax.set_title(f'Transforming Volume (B-scan)', fontsize=24, pad=22)
                plt.imshow(q_t.data.detach().cpu().numpy()[0,0,args.vol_size[0] // 2,:,:], cmap = 'gray', clim=(-1, 1))

                ax = plt.subplot(2,4,3)
                ax.set_title(f'Registered Volume (B-scan)', fontsize=24, pad=22)
                plt.imshow(output.data.detach().cpu().numpy()[0,0,args.vol_size[0] // 2,:,:], cmap = 'gray', clim=(-1, 1))

                ax = plt.subplot(2,4,4)
                ax.set_title(f'Difference (B-scan)', fontsize=24, pad=22)
                plt.imshow(diff, cmap = 'turbo', clim=(0, 2))
                plt.colorbar()

                ax = plt.subplot(2,4,5)
                ax.set_title(f'Reference Volume (en face)', fontsize=24, pad=22)
                plt.imshow(mip_truth[0,0,:,:], cmap = 'gray', clim=(min, max))

                ax = plt.subplot(2,4,6)
                ax.set_title(f'Transforming Volume (en face)', fontsize=24, pad=22)
                plt.imshow(mip_input[0,0,:,:], cmap = 'gray', clim=(min, max))

                ax = plt.subplot(2,4,7)
                ax.set_title(f'Registered Volume (en face)', fontsize=24, pad=22)
                plt.imshow(mip_predicted[0,0,:,:], cmap = 'gray', clim=(min, max))

                ax = plt.subplot(2,4,8)
                ax.set_title(f'Difference (en face)', fontsize=24, pad=22)
                plt.imshow(diff_mip[0,0,:,:], cmap = 'turbo', clim=(0, max - min))
                plt.colorbar()

                plt.tight_layout(pad=4)
                plt.savefig(f'{OUT_PATH}diffusion_timestep_pred_{step + 1}.png')
                plt.close()

        # Save model and EMA weights as .ckpt file to the checkpoint folder
        if is_inference_step:
            torch.save(model.state_dict(), f'{CKPT_PATH}spatial_diffusion_ckpt_{step + 1}.pth')
            torch.save(ema.ema_model.state_dict(), f'{CKPT_PATH}spatial_diffusion_ema_ckpt_{step + 1}.pth')
        
        if is_train_loss_step or is_inference_step:
            torch.save(model.state_dict(), f'{CKPT_PATH}spatial_diffusion_ckpt.pth')
            torch.save(ema.ema_model.state_dict(), f'{CKPT_PATH}spatial_diffusion_ema_ckpt.pth')