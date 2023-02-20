# DTFD-MIL.PyTorch

## Update 2023/01/07

We have updated the pretrained weights of ResNet50 to 'https://download.pytorch.org/models/resnet50-11ad3fa6.pth', and it achieved an AUC of 91.31.

## Update 2023/01/07

After consulting the authors, patch feature extration method in this repo has now been modified.

## Prevoious Version

This repo is built to help readers better understand CVPR2022 paper *DTFD-MIL: Double-tier feature distillation multiple Instance learning for histopathology whole slide image classification*. 

This is NOT an official implementation of the paper. The official one is [here](https://github.com/hrzhang1123/DTFD-MIL)

## What is the difference?

1. Dataset
    - The official one uses a preprocessed pickle file for CAMELYON16 dataset. This file can be downloaded through Google Drive but is pretty huge. 
    - In this repo, we provide the preprocessing code for converting a WSI into patches and then patches into 1024-channel embedding. Therefore, you only need to have the original CAMELYON16 dataset downloaded on your device to train the entire DTFD-MIL framework. However, it should also be noted that we use a different dataset split.

2. Model
    - The official code include three methods for tier-2 distillation, which are MaxMinS, MaxS, and AFS. The experimental result shows that MaxMinS and AFS usually performs well.
    - In this repo, we only have AFS currently.

3. Code-style
    - Compared with the official one, code in this repo is much simpler, which is also easier to understand. We have tested the CAMELYON16 (level 1 patches) on our code but the result shows only an AUC of 86.66% (val) / 82.04% (test).

## How to use?

1. Download CAMELYON16 dataset [here](https://camelyon16.grand-challenge.org/Download/)

2. Clone this repo. Cd into this repo.

3. Run 'Patch-Generation/gen_patch_noLabel_stride_MultiProcessing_multiScales.py' to convert each WSI into a folder of patches. This file is from the official DTFD-MIL repo.

4. Run 'patches2feature.py' to convert each patch into a 1024-dimension vector. This file will run through all the patches in a folder and eventually generate a feats1024.npy for each WSI. In other words, by running this file, we can convert a gigapixel WSI into a Nx1024  matrix (N is the number of patches in a WSI).

5. Run 'train_DTFD-MIL.py' to train and validate the framework.

## Some common problems

- loss1.backward() raises an Exception that gradient has changed. 
    - This is because you use a newer version of PyTorch. Changing to use PyTorch 1.4.0 can solve this problem.

- Prerequests:
    - Pytorch 1.4.0
    - scikit-learn (or not)
    - numpy
    - opencv


