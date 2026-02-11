# Spatial Diffusion: A Scalable Self-Supervised Deep Learning Framework for Robust 3D Medical Imaging Registration

<img width="690" height="328" alt="image" src="https://github.com/user-attachments/assets/5eb76e91-d3db-4587-ae3c-9c52e8d072d4" />

Spatial diffusion is a novel, self-supervised deep learning framework for 3D affine registration that is inspired by the success of Denoising Diffusion Probabilistic Models. Registration is decomposed into a series of small affine transformations determined by a Gaussian diffusion process, enabling fully self-supervised training without intensity-based similarity metrics.

---

## Table of Contents
- [Installation](#installation)  
- [Dataset and Pre-trained Models](#dataset)
- [Overview](#overview)
- [Usage](#usage)  
  - [Training](#training)  
  - [Inference](#inference)  

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
Spatial diffusion was evaluated on optical coherence tomography (OCT), OCT angiography, T1-weighted brain MRI, and T2-weighted head and neck MRI of venous malformations. However, only the OASIS-1 dataset of T1-weighted brain MRI is publicly available, which can be downloaded (already preprocessed) at https://github.com/adalca/medical-datasets/blob/master/neurite-oasis.md.

Additionally, the MNI152 brain template can be found at https://github.com/cwmok/C2FViT/tree/main/Data (MNI152_T1_1mm_brain_pad_RSP.nii.gz is the volume of the brain and image_A_seg4_mni.nii.gz is the segmentation mask).

### Model Weights
The model weights can be installed at:
