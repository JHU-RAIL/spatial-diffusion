import torch
from torch.utils.data.dataset import Dataset
from typing import Tuple, List, Union, Callable, Any, Optional
import numpy as np
from scipy.ndimage import zoom
from skimage import exposure
import nibabel as nib
import multiprocessing as mp
from tqdm import tqdm
import warnings
import pickle
import random
import os

class MultimodalNIFTIDataset3D():
    def __init__(
        self,
        file_lists: List[List[str]],
        volume_size: Optional[Tuple[int, int, int]] = None,
        mean: float = 0.5,
        std: float = 0.5,
        types: List[str] = None,
        batch_size: int = 1,
        num_workers: int = 8,
        shuffle: bool = False,
        persistent_workers: bool = False
    ) -> None:
        """
        General dataset class that loads a set of nifti files,
        normalizing the scans by a mean and standard deviation.
        """
        # Number of datasets should match the number of specified types
        if types is not None and len(file_lists) != len(types):
            raise Exception(f'Expected a list of {len(file_lists)} dataset types, but got {len(types)}!')
        
        self.dataloaders = []   # List of dataloaders
        self.loader_queue = []  # Paired dataloader and dataset type queue
        self.types = [] # Dataset types 
        self.size = 0   # Total batches across all dataloaders

        for i, files in enumerate(file_lists):
            print(f'\nProcessing dataset [{i+1}/{len(file_lists)}].')

            # If no dataset types are specified, simply default to 'simple' (no special preprocessing)
            type = 'simple' if types is None else types[i].lower()
            self.types.append(type)

            # Create datasets from the list of files
            dataset = NIFTIDataset3D(files, volume_size, mean, std, type, i, num_workers)

            # Wrap datasets in dataloaders
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, 
                persistent_workers=persistent_workers
            )

            # Update load of dataloaders and total dataset size
            self.dataloaders.append(dataloader)
            self.loader_queue.append((iter(dataloader), types))
            self.size += len(dataloader)

        self.num_loaders = len(self.dataloaders)    # Number of dataloaders
        self.shuffle = shuffle  # Whether to shuffle the dataset

    def __iter__(self) -> 'MultimodalNIFTIDataset3D':
        """
        Returns an iterator for the dataset.
        """
        self.reset()
        return self  # Return the dataset object itself as an iterator

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a volume from the dataset and its file name.
        """
        # NOTE: Current implementation is O(N) worst-case complexity. There is almost
        # certainly a way to improve efficiency, but the general the expectation is 
        # that N (number of dataset modalities) is relatively small.

        i = 0
        while i < self.num_loaders:
            # Retrieve element from back of loader queue
            loader, type = self.loader_queue.pop()

            if self.shuffle:
                # Re-add dataloader at a random position in the loader queue
                rand_idx = random.randint(0, self.num_loaders - 1)
                self.loader_queue.insert(rand_idx, (loader, type))
            else:
                # Re-add element to the front, following a standard round-robin scheme
                self.loader_queue.insert(0, (loader, type))

            try:
                # Retrieve the next element from the dataloader
                vol, filename, v_range, class_id = next(loader)
                return vol, v_range, type, class_id
            except StopIteration:
                # Dataloader has reached the end -- skip and move to the next one
                i += 1

        # All dataloaders in the queue have reached the end
        raise StopIteration

    def __len__(self) -> int:
        """
        Returns the size of the dataset.
        """
        return self.size

    def reset(self) -> None:
        """
        Resets the dataloader iterators.
        """
        # Loop through all dataloaders and re-initialize their iterators
        for i in range(self.num_loaders):
            self.loader_queue[i] = (iter(self.dataloaders[i]), self.types[i])

class NIFTIDataset3D(Dataset):
    def __init__(
        self,
        files: List[str],
        volume_size: Optional[Tuple[int, int, int]] = None,
        mean: float = 0.5,
        std: float = 0.5,
        type: str = 'simple',
        class_id: Optional[int] = None,
        num_workers: int = 8
    ) -> None:
        """
        General dataset class that loads a set of nifti files,
        normalizing the scans by a mean and standard deviation.
        """
        # Store variables
        self.size = len(files)
        self.volume_size = volume_size
        self.mean = mean
        self.std = std
        self.class_id = class_id
        self.volumes = []

        max_workers = mp.cpu_count()

        # Check that requested number of workers are available
        if max_workers < num_workers:
            raise Exception(f'Requested {num_workers} workers, but only {max_workers} CPU cores are available!')

        # Warn user if requested number of workers are close to the maximum number of cores
        if num_workers >= max_workers - 1:
            warnings.warn(f'Warning: The requested {num_workers} workers is close to the maximum {max_workers} '\
                'available CPU cores. Decreasing the number of workers is highly recommended.')

        self.type = type.lower()

        # Select the appropriate data loader
        if self.type == 'oct':
            # Use OCT dataloader (pad + crops above and below retina)
            print(f'\nLoading {self.size} OCT files with {num_workers} worker(s)...')
            loader = self._oct_loader

        elif self.type == 'octa':
            # Use OCTA dataloader (pad + crops above and below retina)
            print(f'\nLoading {self.size} OCTA files with {num_workers} worker(s)...')
            loader = self._octa_loader

        elif self.type == 'mri':
            # Use MRI dataloader (pad + no special preprocessing)
            print(f'\nLoading {self.size} MRI files with {num_workers} worker(s)...')
            loader = self._mri_loader

        elif self.type == 'venous':
            # Use venous malformation dataloader (pad + resample to isotropic voxels)
            print(f'\nLoading {self.size} venous malformation files with {num_workers} worker(s)...')
            loader = self._venous_loader

        elif self.type == 'simple':
            # Use simple, general-purpose dataloader (no special preprocessing)
            print(f'\nLoading {self.size} files with {num_workers} worker(s)...')
            loader = self._simple_loader
        
        elif self.type == 'simple_iso':
            # Use simple, general-purpose dataloader with resampling to isotropic voxels
            print(f'\nLoading {self.size} files with {num_workers} worker(s)...')
            loader = self._simple_isotropic_loader

        else:
            # Unknown dataset type
            raise ValueError(f'Unknown dataset type. Expected \'oct\', \'octa\', '\
                f'\'mri\', \'venous\', \'simple\', or \'simple_iso\', but got \'{self.type}\'!')

        # Set multiprocessing method to 'spawn' (appears to be MUCH faster for preprocessing)
        default_mp_method = mp.get_start_method()
        mp.set_start_method('spawn', force=True)

        # Set number of workers for CPU multi-threading and load the volumes into memory
        with mp.Pool(num_workers) as pool:
            self.volumes = list(tqdm(pool.imap(loader, files), total=len(files)))

        # Set multiprocessing method back to the default after finished
        mp.set_start_method(default_mp_method, force=True)
        print('Files loaded successfully!\n')

    def __getitem__(self, index: int) -> Union[Tuple[torch.Tensor, str], Tuple[torch.Tensor, str, int]]:
        """
        Returns a volume from the dataset, its file name, and class ID if specified.
        """
        # Check that index is not out of bounds
        if index >= len(self): raise IndexError

        # Return the volumes
        if self.class_id is None:
            return self.volumes[index]
        else:
            return (*self.volumes[index], self.class_id)

    def __len__(self) -> int:
        """
        Returns the size of the dataset.
        """
        return self.size

    def _oct_loader(self, file_path: str) -> Tuple[np.ndarray, str]:
        """
        Loads an OCT retinal imaging nifti file. Performs
        cropping of dark regions above and below the retina
        and pads volume to target dimensions.
        """
        return self._retinal_loader(file_path, crop_tol=0.8 * 255)

    def _octa_loader(self, file_path: str) -> Tuple[np.ndarray, str]:
        """
        Loads an OCTA retinal imaging nifti file. Performs
        cropping of dark regions above and below the retina
        and pads volume to target dimensions.
        """
        return self._retinal_loader(file_path, crop_tol=0.7 * 255)

    def _retinal_loader(self, file_path: str, crop_tol: float = 0.7 * 255) -> Tuple[np.ndarray, str]:
        """
        Loads an OCT/OCTA retinal imaging nifti file. Performs
        cropping of dark regions above and below the retina and
        pads volume to target dimensions.
        """
        # Load OCT/OCTA volume from file
        volume = nib.load(file_path).get_fdata()
        volume = np.flip(np.flip(volume, 1), 2)

        # Crop dark regions above and below the retina
        volume, _ = NIFTIDataset3D._crop_volume(volume.copy().astype(np.float32), self.volume_size, tol=crop_tol)

        # Normalize volume pixel intensities between 0 - 1 and pad the volume
        v_min, v_max = volume.min(), volume.max()
        volume = (volume - v_min) / (v_max - v_min)
        volume = np.stack([self._pad(volume, self.volume_size, maintain_aspect=True)], axis=0)

        # Normalize pixel intensities by mean and standard deviation
        volume = (volume - self.mean) / self.std

        return torch.from_numpy(volume), file_path, torch.from_numpy(np.array([v_min, v_max], dtype=np.float32))

    def _mri_loader(self, file_path: str) -> Tuple[np.ndarray, str]:
        """
        Loads an MRI nifti file and pads volume to the
        target dimension.
        """
        # Load 3D volume from file
        volume = nib.load(file_path).get_fdata().copy().astype(np.float32)

        # Normalize volume pixel intensities between 0 - 1 and pad the volume
        v_min, v_max = volume.min(), volume.max()
        volume = (volume - v_min) / (v_max - v_min)
        volume = np.stack([self._pad(volume, self.volume_size, maintain_aspect=True)], axis=0)

        # Normalize pixel intensities by mean and standard deviation
        volume = (volume - self.mean) / self.std

        return torch.from_numpy(volume), file_path, torch.from_numpy(np.array([v_min, v_max], dtype=np.float32))

    def _venous_loader(self, file_path: str) -> Tuple[np.ndarray, str]:
        """
        Loads a venous malformation MRI nifti file, resamples to
        isotropic voxels, and resizes + pads the volume to the 
        target dimension.
        """
        # Load 3D volume from file
        nifti = nib.load(file_path)
        affine = nifti.affine
        volume = nifti.get_fdata().copy().astype(np.float32)

        # Compute physical voxel dimensions
        vox_dim = np.linalg.norm(affine[:3,:3], axis=0)

        # Rearrange dimensions from (x, y, z) --> (d, h, w)
        volume = np.transpose(volume, axes=(2, 1, 0))

        # Compute scale factor for isotropic voxel dimensions
        base_vox_size = np.min(vox_dim)

        if base_vox_size <= 0:
            raise ValueError(f'Volume at file path \'{file_path}\' contains an invalid affine matrix. ' \
                             'Physical voxel dimensions must be positive.')

        scale_factors = np.array([
            abs(vox_dim[2] / base_vox_size),
            abs(vox_dim[1] / base_vox_size),
            abs(vox_dim[0] / base_vox_size)
        ])

        # Adjust scale factor to match output volume dimensions with the target
        if self.volume_size is not None:
            target_size_scale = np.max(np.array(volume.shape) * scale_factors / np.array(self.volume_size))
            scale_factors /= target_size_scale
        else:
            raise ValueError('Volume size must be defined for the venous malformation dataloader!')

        # Resample volume using the computed scale factor across each dimensions
        volume = zoom(volume, scale_factors)

        # Normalize volume pixel intensities between 0 - 1 and pad the volume
        v_min, v_max = volume.min(), volume.max()
        volume = (volume - v_min) / (v_max - v_min)
        volume = exposure.equalize_adapthist(volume, kernel_size=(32, 32, 32), clip_limit=0.05)
        volume = np.stack([self._pad(volume, self.volume_size, maintain_aspect=True)], axis=0)

        # Normalize pixel intensities by mean and standard deviation
        volume = (volume - self.mean) / self.std

        return torch.from_numpy(volume), file_path, torch.from_numpy(np.array([v_min, v_max], dtype=np.float32))

    def _simple_loader(self, file_path: str) -> Tuple[np.ndarray, str]:
        """
        Loads a nifti file without any special preprocessing.
        """
        # Load 3D volume from file
        volume = nib.load(file_path).get_fdata().copy().astype(np.float32)
        volume = np.stack([volume], axis=0)

        # Normalize volume pixel intensities between 0 - 1
        v_min, v_max = volume.min(), volume.max()
        volume = (volume - v_min) / (v_max - v_min)

        # Normalize pixel intensities by mean and standard deviation
        volume = (volume - self.mean) / self.std

        return torch.from_numpy(volume), file_path, torch.from_numpy(np.array([v_min, v_max], dtype=np.float32))

    def _simple_isotropic_loader(self, file_path: str) -> Tuple[np.ndarray, str]:
        """
        Loads a nifti file and resamples to isotropic voxels.
        """
        # Load 3D volume from file
        nifti = nib.load(file_path)
        affine = nifti.affine
        volume = nifti.get_fdata().copy().astype(np.float32)

        # Compute physical voxel dimensions
        vox_dim = np.linalg.norm(affine[:3,:3], axis=0)

        # Rearrange dimensions from (x, y, z) --> (d, h, w)
        volume = np.transpose(volume, axes=(2, 1, 0))

        # Compute scale factor for isotropic voxel dimensions
        base_vox_size = np.min(vox_dim)

        if base_vox_size <= 0:
            raise ValueError(f'Volume at file path \'{file_path}\' contains an invalid affine matrix. ' \
                             'Physical voxel dimensions must be positive.')

        scale_factors = np.array([
            abs(vox_dim[2] / base_vox_size),
            abs(vox_dim[1] / base_vox_size),
            abs(vox_dim[0] / base_vox_size)
        ])

        # Resample volume using the computed scale factor across each dimensions
        volume = zoom(volume, scale_factors)

        # Normalize volume pixel intensities between 0 - 1
        v_min, v_max = volume.min(), volume.max()
        volume = (volume - v_min) / (v_max - v_min)
        volume = np.stack([volume], axis=0)

        # Normalize pixel intensities by mean and standard deviation
        volume = (volume - self.mean) / self.std
        return torch.from_numpy(volume), file_path, torch.from_numpy(np.array([v_min, v_max], dtype=np.float32))

    def _pad(
        self,
        vol: np.ndarray,
        target_size: Tuple[int, int, int],
        pad_value: float = 0.0,
        maintain_aspect: bool = True
    ) -> np.ndarray:
        """
        Pads 3D volume to a fixed target size. If maintain_aspect is true,
        volumes with input dimensions larger than the target size will be
        padded to dimension of the same aspect ratio.
        """
        # Get the current shape of the input volume
        D, H, W = vol.shape

        # Maintain aspect ratio of the target dimensions if input dimensions are larger than target
        target_dim = target_size
        if maintain_aspect:
            # Compute dimensions with depth as the fixed reference
            if D > target_size[0]:
                dim = (D, D * target_size[1] // target_size[0], D * target_size[2] // target_size[0])
                target_dim = dim if dim > target_dim else target_dim

            # Compute dimensions with height as the fixed reference
            if H > target_size[1]:
                dim = (H * target_size[0] // target_size[1], H, H * target_size[2] // target_size[1])
                target_dim = dim if dim > target_dim else target_dim

            # Compute dimensions with width as the fixed reference
            if W > target_size[2]:
                dim = (W * target_size[0] // target_size[2], W * target_size[1] // target_size[2], W)
                target_dim = dim if dim > target_dim else target_dim

        # Calculate the padding needed for each dimension
        pad_D = max(0, target_dim[0] - D)
        pad_H = max(0, target_dim[1] - H)
        pad_W = max(0, target_dim[2] - W)
        
        # Calculate the padding for each side (before and after)
        pad_D_before = pad_D // 2
        pad_D_after = pad_D - pad_D_before
        pad_H_before = pad_H // 2
        pad_H_after = pad_H - pad_H_before
        pad_W_before = pad_W // 2
        pad_W_after = pad_W - pad_W_before
        padding = ((pad_D_before, pad_D_after), (pad_H_before, pad_H_after), (pad_W_before, pad_W_after))
        
        # Pad the volume
        padded_volume = np.pad(vol, padding, mode='constant', constant_values=pad_value)
        return padded_volume

    @staticmethod
    def _crop_volume(
        vol: np.ndarray,
        volume_size: Tuple[int, int, int],
        tol: float = 0.0,
        crop_pos: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Crops dark regions above and below the tissue. Intended for
        retinal 3D OCT/OCTA volumes.
        """
        # Nothing to crop if target volume size is not defined
        if volume_size is None:
            return vol

        if crop_pos is None:
            height = vol.shape[1]

            # Mask of non-black pixels (assuming image has a single channel).
            mask = vol > tol

            # Coordinates of non-black pixels.
            coords = np.argwhere(mask)

            # Bounding box of non-black pixels.
            x0, y0, z0 = coords.min(axis=0)

            if y0 + volume_size[1] > height:
                y0 = height - volume_size[1]
        else:
            y0 = crop_pos

        # Get the contents of the bounding box.
        cropped = vol[:,y0:y0+volume_size[1],:]
        return cropped, y0

class DatasetLoader():
    def __init__(self, val_split: float = 0.1):
        """
        Helper class for loading and writing the training and
        validation splits. The training dataset will be split based on
        the validation proportion if no validation data is provided.
        """
        self.val_split = val_split
        self.train_files = None
        self.val_files = None

    def load(self, train_path: str, val_path: Union[str, None]) -> Tuple[List[Any], List[Any]]:
        """
        Loads training and validation splits from a folder or
        a pickled file. Training dataset is split according to
        the validation proportion if val_path is not specified.
        """
        if os.path.isdir(train_path):
            # Retrieve a list of files from the folder of training files
            self.train_files = [os.path.join(train_path, f) for f in os.listdir(train_path)]
        else:
            # Unpickle the file containing training files
            with open(train_path, 'rb') as f:
                self.train_files = pickle.load(f)

            print('Unpickled training splits.')

        # No validation files provided -- split the training dataset
        if val_path is None:
            print(f'No validation path specified for \'{train_path}\'. Splitting training dataset...')
            self.train_files, self.val_files = self.split(self.train_files)

            return self.train_files, self.val_files

        if os.path.isdir(val_path):
            # Retrieve a list of files from the folder of validation files
            self.val_files = [os.path.join(val_path, f) for f in os.listdir(val_path)]
        else:
            # Unpickle the file containing validation files
            with open(val_path, 'rb') as f:
                self.val_files = pickle.load(f)
            
            print('Unpickled validation splits.')

        return self.train_files, self.val_files

    def split(self, dataset: List[Any]) -> Tuple[List[Any], List[Any]]:
        """
        Split a list into training and validation datasets.
        """
        # Shuffle the dataset files
        random.shuffle(dataset)

        # Calculate split index based on validation proportion
        split_index = int(len(dataset) * self.val_split)

        # Split the list into train and validation files
        train_files = dataset[split_index:]
        val_files = dataset[:split_index]

        return train_files, val_files

    def write(self, train_path: str, val_path: str) -> None:
        """
        Write training and validation splits as a pickle file.
        """
        if self.train_files is None or self.val_files is None:
            raise Exception('Load a training and validation dataset with load() first!')

        # Get parent directory of the training and validation split files
        train_dir = os.path.dirname(train_path)
        val_dir = os.path.dirname(val_path)

        # Create output directories if necessary
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)

        if not os.path.exists(val_dir):
            os.makedirs(val_dir)

        # Save training and validation splits as a pickle file
        with open(train_path, 'wb') as f:
            pickle.dump(self.train_files, f)

        with open(val_path, 'wb') as f:
            pickle.dump(self.val_files, f)
