# Spatial Diffusion: A Scalable Self-Supervised Deep Learning Framework for Robust 3D Medical Imaging Registration

<img width="690" height="328" alt="image" src="https://github.com/user-attachments/assets/5eb76e91-d3db-4587-ae3c-9c52e8d072d4" />

This is the author's official PyTorch implementation of "Spatial Diffusion: A Scalable Self-Supervised Deep Learning Framework for Robust 3D Medical Imaging Registration." Spatial diffusion is a novel, self-supervised deep learning framework for 3D affine registration that is inspired by the success of Denoising Diffusion Probabilistic Models. Registration is decomposed into a series of small affine transformations determined by a Gaussian diffusion process, enabling fully self-supervised training without intensity-based similarity metrics.

---

## Table of Contents
- [Installation](#installation)
  - [Requirements](#requirements)  
  - [Setup](#setup) 
- [Dataset and Pre-trained Models](#dataset-and-pre-trained-models)
  - [Data](#data)  
  - [Model Weights](#model-weights) 
- [Overview](#overview)
  - [Diffusion](#diffusion)  
  - [Utilities](#utilities)
  - [Scripts](#scripts) 
- [Usage](#usage)  
  - [Training](#training)  
  - [Inference](#inference)
- [Citation](#citation)

---

## Installation
### Requirements
Install PyTorch 1.8.0 (or newer), along with additional dependencies (though this implementation was tested on PyTorch 2.7.0). The full list of required dependencies can be found in `requirements.txt` and are listed below:
```
ema_pytorch
fvcore
matplotlib
nibabel
numpy
scikit_image
scipy
torch
tqdm
```

### Setup
To use, clone the repository and install the dependencies in `requirements.txt`.
```bash
# Clone repository
git clone https://github.com/JHU-RAIL/spatial-diffusion.git
cd spatial-diffusion

# Install dependencies
pip install -r requirements.txt
```

## Dataset and Pre-trained Models
### Data
Spatial diffusion was evaluated on optical coherence tomography (OCT), OCT angiography (OCTA), T1-weighted brain MRI, and T2-weighted head and neck MRI of venous malformations. However, only the OASIS-1 dataset of T1-weighted brain MRI is publicly available, which can be downloaded (already preprocessed) at https://github.com/adalca/medical-datasets/blob/master/neurite-oasis.md.

Additionally, the MNI152 brain template can be found at https://github.com/cwmok/C2FViT/tree/main/Data (`MNI152_T1_1mm_brain_pad_RSP.nii.gz` is the volume of the brain and `image_A_seg4_mni.nii.gz` is the segmentation mask).

### Model Weights
The model weights can be downloaded at:

Coming soon. Stay tuned!

If you use our pre-trained models, please refer to the [citation section](#citation) for guidance on how to cite our work.

## Overview
### Diffusion
`spatial_diffusion.py` - [implementation of the 3D spatial diffusor](https://github.com/JHU-RAIL/spatial-diffusion/blob/main/diffusion/spatial_diffusion.py), which includes random sampling of the forward diffusion process for training and prediction of the full reverse diffusion process for inferencing.

`spatial_diffusion_model.py` - [implementation of the 3D Affine Spatial Transformer model](https://github.com/JHU-RAIL/spatial-diffusion/blob/main/diffusion/spatial_diffusion_model.py) as a CNN, which predicts affine transformations given a pair of input volumes. The implementation includes preprocessing, time embeddings, and modality-conditional embeddings.

### Utilities
`augmentations.py` - [implementation of synthetic 3D artifacts and augmentations](https://github.com/JHU-RAIL/spatial-diffusion/blob/main/utils/augmentations.py) for self-supervised training of spatial diffusion. Includes stretch, shadowing, motion, doubling, and deformation artifacts, as well as random crop and random rigid-body transformation augmentations.

`datasets.py` - [implementation of a multi-modal 3D dataloader](https://github.com/JHU-RAIL/spatial-diffusion/blob/main/utils/datasets.py) that performs normalization and standard preprocessing for OCT, OCTA, T1-weighted brain MRI, and T2-weighted head and neck MRI provided as NIFTI files. Automatically assigns modality-specific class IDs for multimodal training of a conditional spatial diffusion model.

### Scripts
`train_spatial_diffusion.py` - [script for training spatial diffusion](https://github.com/JHU-RAIL/spatial-diffusion/blob/main/train_spatial_diffusion.py), supporting both unimodality and multimodality training.

`inference_spatial_diffusion.py` - [script for inferencing spatial diffusion](https://github.com/JHU-RAIL/spatial-diffusion/blob/main/inference_spatial_diffusion.py), registering the specified moving volumes to the reference volumes provided as NIFTI files. Supports unconditional, modality-condition, and zero-shot cross-modality registration with spatial diffusion.

## Usage
### Training
To train spatial diffusion on a single modality:

```bash
python3 train_spatial_diffusion.py --train ./path/to/dataset/ --dataset [oct/octa/mri/venous] --save_path ./results/unimodality --gpu 0 --workers 12 --steps 115000
```

To train spatial diffusion on multiple modalities:

```bash
python3 train_spatial_diffusion.py --train ./path/to/dataset1/ ./path/to/dataset2/ ./path/to/dataset3/ ./path/to/dataset4/ --val - - ./path/to/dataset3/validation/ - --dataset [oct/octa/mri/venous] [oct/octa/mri/venous] [oct/octa/mri/venous] [oct/octa/mri/venous] --save_path ./results/foundation --gpu 0 --workers 12 --steps 265000
```

The model weights will be saved in `/ckpt` within the folder specified by `--save_path`. Both the exponential moving average (EMA) and non-EMA model weights are saved, where the `spatial_diffusion_ema_ckpt.pth` file name refers to the EMA model and `spatial_diffusion_ckpt.pth` refers to the non-EMA model.

### Inference
To inference an unconditional spatial diffusion model trained on a single modality:

```bash
python3 inference_spatial_diffusion.py --ref ../path/to/scan.nii.gz --mov ../path/to/moving/scans/*.nii.gz --ckpt ./path/to/model/ckpts/spatial_diffusion_ema_ckpt_XXXXXX.pth --dataset [oct/octa/mri/venous] --bidir --gpu 0
```

To inference a modality-conditional spatial diffusion model trained on multiple modalities:

```bash
python3 inference_spatial_diffusion.py --ref ../path/to/scan.nii.gz --mov ../path/to/moving/scans/*.nii.gz --ckpt ./path/to/model/ckpts/spatial_diffusion_ema_ckpt_XXXXXX.pth --dataset [oct/octa/mri/venous] --n_classes 4 --cond_class 2 --bidir --gpu 0
```

To perform zero-shot cross-modality registration, set `--cond_class 0` on a model previously trained on multiple modalities. This is also useful for foundation models performing intra-modality registration on modalities with few training samples.

### Citation

If you find this repository useful, please cite our work:

- Sumpena, E., Cornelio, A., Seshadri, S., Beiser, A., Wang, D., Ringman, J., Kashani, A., Jones, C., **“Spatial Diffusion: A Self-Supervised Deep Learning Algorithm for Artifact-Resilient 3D OCTA Registration,”** Presented at the Association for Research in Vision and Ophthalmology (ARVO) Annual Meeting, May 2025.
