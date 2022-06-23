# Towards Online Real-Time Memory-based Video Inpainting Transformers

## Introduction

## Install

### Packages environment

### Data organization

You will need to add two folders to be able to run the scripts: checkpoints and data

First, create the folders
```
mkdir checkpoints
mkdir data
```
In 'checkpoints' put the pretrained weights for the different models, available [here](https://drive.google.com/drive/folders/1hVR2y9ahJu2tt7zIfSUiMg7fK5q3IJOH?usp=sharing).

In 'data' put the frames and masks of the videos you want to inpaint in distinct folders. Each subfolder represents a video, and the name should match between the frames and the masks.

A suggested organization is the following:
```
checkpoints
    |DSTT.pth
    |FuseFormer.pth
    |E2FGVI.pth
    |i3d_rgb_imagenet.pt
    
data
    |DAVIS
        |JPEGImages
            |bear
                |00000.jpg
                |00001.jpg
                ...
            |blackswan
                |00000.jpg
                ...
            ...
        |Masks
            |bear
                |00000.jpg
                |00001.jpg
                ...
            |blackswan
                |00000.jpg
                ...
            ...
    |YTVOS
        |JPEGImages
            ...
        |Masks
            ...
```

### DAVIS and YouTube-VOS datasets



## Usage

You can easily run any online inpainting script with the command:

```
python evaluate_[backbone]_[model].py --ckpt checkpoints/[backbone].pth --video data/DAVIS/JPEGImages --mask data/DAVIS/Masks --evaluate --save_results
```
- [backbone] is the name of the original inpainting algorithm: DSTT / FuseFormer / E2FGVI
- [model] is the name of the online inpainting pipeline to use: O (Online) / OM (Online-Memory) / OMR (Online-Memory-Refinement)
- --video is the path of the corrupted images (see Data Organization)
- --mask is the path of the masks for these images
- --evaluate means that the script will evaluate the inpaintings on different metrics (PSNR/SSIM/VFID)
- --save_results means that the inpaintings will be saved (as a folder of images)

Other parameters can be added to the command:
- --save_folder to specify where to save the resulting inpaintings
- --num_neighbors to specify the number of neighboring frames to use for the prediction
- --ref_step and --num_refs to specify the number and frequency of the reference frames for the prediction

Examples:
```
python evaluate_DSTT_O.py --ckpt checkpoints/DSTT.pth --video data/DAVIS/JPEGImages --mask data/DAVIS/Masks --evaluate 
python evaluate_E2FGVI_OM.py --ckpt checkpoints/E2FGVI.pth --video data/YTVOS/JPEGImages --mask data/YTVOS/Masks --save_results
python evaluate_FuseFormer_OMR.py --ckpt checkpoints/FuseFormer.pth --video data/my_dataset/Images --mask data/my_dataset/Masks 
```
