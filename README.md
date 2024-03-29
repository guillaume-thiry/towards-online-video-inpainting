# Towards Online Real-Time Memory-based Video Inpainting Transformers

## Introduction

[To be done]

## Install

### Packages environment

The Conda environment can easily be installed using the file *env.yml*:

```
conda env create --file env.yml
conda activate online-inpainting
```

This environement is very close to the one provided in [E2FGVI](https://github.com/MCG-NKU/E2FGVI):

- Python $\ge$ 3.7
- Torch $\ge$ 1.5
- CUDA $\ge$ 10.1

### Data organization

You will need to add two folders to be able to run the scripts: 'checkpoints' and 'data'.

First, create the folders:
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
                |00000.png
                |00001.png
                ...
            |blackswan
                |00000.png
                ...
            ...
    |YTVOS
        |JPEGImages
            ...
        |Masks
            ...
```

### DAVIS and YouTube-VOS datasets

To reproduce the results of the paper on DAVIS and YouTube-VOS, download the datasets following these instructions:

- DAVIS dataset available [here](https://davischallenge.org/davis2017/code.html) (Semi-supervised TrainVal - 90 videos)
- YouTube-VOS available [here](https://competitions.codalab.org/competitions/20127) (test_all_frames - 541 videos)

Note: For YouTube-VOS, you will need to rename the files so that each video starts at 00000.jpg (to match the masks).

For the masks, we extended the set created by [FuseFormer](https://github.com/ruiliu-ai/FuseFormer) to include more DAVIS videos (from 50 to 90). They are available [here](https://drive.google.com/drive/folders/1n1Rg4L5TZnoz1Vjk_EC_TgSrs18m_vNU?usp=sharing).


## Usage

### Commands

You can easily run any online inpainting script with the command:

```
python evaluate_[backbone]_[model].py --ckpt checkpoints/[backbone].pth --video data/DAVIS/JPEGImages --mask data/DAVIS/Masks --evaluate --save_results
```
- *[backbone]* is the name of the original inpainting algorithm: DSTT / FuseFormer / E2FGVI
- *[model]* is the name of the online inpainting pipeline to use: O (Online) / OM (Online-Memory) / OMR (Online-Memory-Refinement)
- *--video* is the path of the corrupted images (see Data Organization)
- *--mask* is the path of the masks for these images
- *--evaluate* means that the script will evaluate the inpaintings on different metrics (PSNR/SSIM/VFID)
- *--save_results* means that the inpaintings will be saved (as a folder of images)

Other parameters can be added to the command:
- *--save_folder* to specify where to save the resulting inpaintings
- *--num_neighbors* to specify the number of neighboring frames to use for the prediction
- *--ref_step* and *--num_refs* to specify the number and frequency of the reference frames for the prediction

Examples:
```
python evaluate_DSTT_O.py --ckpt checkpoints/DSTT.pth --video data/DAVIS/JPEGImages --mask data/DAVIS/Masks --evaluate 
python evaluate_E2FGVI_OM.py --ckpt checkpoints/E2FGVI.pth --video data/YTVOS/JPEGImages --mask data/YTVOS/Masks --save_results
python evaluate_FuseFormer_OMR.py --ckpt checkpoints/FuseFormer.pth --video data/my_dataset/Images --mask data/my_dataset/Masks 
```

### Metrics

To easily obtain the average PSNR, SSIM, and VFID of the videos, just add the argument *--evaluate* in the commands above. 

To evaluate the warping error (Ewarp), you will first need to save the inpaited frames using the argument *--save_results* in the commands. The metric can then be calculated following instructions from [video-inpainting-evaluation](https://github.com/MichiganCOG/video-inpainting-evaluation/tree/public) to retrieve the values given in the paper. Another possibility is to follow [fast-blind-video-consistency](https://github.com/phoenix104104/fast_blind_video_consistency) but the results may differ from the paper as they depend on the way to calculate the flows.

## Citing this work

[To be done]

## Acknowledgement

Our work is built upon three existing video inpainting models: **DSTT**, **FuseFormer** and **E2FGVI**. In particular, our code shares a lot with these existing works.

- [DSTT](https://github.com/ruiliu-ai/DSTT) (Liu et al. 2021)
- [FuseFormer](https://github.com/ruiliu-ai/FuseFormer) (Liu et al. 2021)
- [E2FGVI](https://github.com/MCG-NKU/E2FGVI) (Li et al. 2022)
