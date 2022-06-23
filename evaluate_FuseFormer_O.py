# -*- coding: utf-8 -*-
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import importlib
import os
import argparse
import copy
import random
import sys
import json
from skimage import measure
from skimage import metrics
from core.utils import create_random_shape_with_random_motion
import time

import torch
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models
import torch.multiprocessing as mp
from torchvision import transforms

# My libs
from core.utils import Stack, ToTorchFormatTensor
from model.i3d import InceptionI3d
from scipy import linalg


parser = argparse.ArgumentParser(description='FuseFormer_O')
parser.add_argument("--video", type=str, required=True)
parser.add_argument("--mask", type=str, required=True)
parser.add_argument("--ckpt", type=str, default='checkpoints/FuseFormer.pth')
parser.add_argument("--width", type=int, default=432)
parser.add_argument("--height", type=int, default=240)
parser.add_argument("--ref_step", type=int, default=10)
parser.add_argument("--num_neighbors", type=int, default=3)
parser.add_argument("--num_refs", type=int, default=3)
parser.add_argument("--save_results", action='store_true')
parser.add_argument("--save_folder", type=str, default='results')
parser.add_argument("--evaluate", action='store_true')
args = parser.parse_args()


w, h = args.width, args.height
ref_step = args.ref_step
num_neighbors = args.num_neighbors
num_refs = args.num_refs
i3d_model = None

_to_tensors = transforms.Compose([
    Stack(),
    ToTorchFormatTensor()])

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representive data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representive data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +  # NOQA
            np.trace(sigma2) - 2 * tr_covmean)

def get_fid_score(real_activations, fake_activations):
    """
    Given two distribution of features, compute the FID score between them
    """
    m1 = np.mean(real_activations, axis=0)
    m2 = np.mean(fake_activations, axis=0)
    s1 = np.cov(real_activations, rowvar=False)
    s2 = np.cov(fake_activations, rowvar=False)
    return calculate_frechet_distance(m1, s1, m2, s2)

def init_i3d_model():
    global i3d_model
    if i3d_model is not None:
        return

    i3d_model_weight = './checkpoints/i3d_rgb_imagenet.pt'
    i3d_model = InceptionI3d(400, in_channels=3, final_endpoint='Logits')
    i3d_model.load_state_dict(torch.load(i3d_model_weight))
    i3d_model.to(torch.device('cuda:0'))

def get_i3d_activations(batched_video, target_endpoint='Logits', flatten=True, grad_enabled=False):
    """
    Get features from i3d model and flatten them to 1d feature,
    valid target endpoints are defined in InceptionI3d.VALID_ENDPOINTS
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7', 'MaxPool3d_2a_3x3', 'Conv3d_2b_1x1', 'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3', 'Mixed_3b', 'Mixed_3c', 'MaxPool3d_4a_3x3',
        'Mixed_4b', 'Mixed_4c', 'Mixed_4d', 'Mixed_4e', 'Mixed_4f',
        'MaxPool3d_5a_2x2', 'Mixed_5b', 'Mixed_5c', 'Logits', 'Predictions',
    )
    """
    init_i3d_model()
    with torch.set_grad_enabled(grad_enabled):
        feat = i3d_model.extract_features(batched_video.transpose(1, 2), target_endpoint)
    if flatten:
        feat = feat.view(feat.size(0), -1)

    return feat


def get_ref_index(f, neighbor_ids, ref_step, num_refs):
    """
    Returns the last reference frames before the current frames not in its neighborhood
    ref_steps = sampling factor (10 by default)
    num_refs = number of frames to return
    f = current frame
    neighbor_ids = neighborhood of the current frame
    """
    ref_index = []
    i = f-1
    while(i>=0 and len(ref_index)<num_refs):
        if not i in neighbor_ids and not i%ref_step:
            ref_index.append(i)
        i -= 1
    return ref_index[::-1]

def get_videos_list(args):
    video_dir = args.video
    video_folder = sorted(os.listdir(video_dir))
    video_list = [os.path.join(video_dir,f) for f in video_folder]

    mask_dir = args.mask
    mask_folder = sorted(os.listdir(mask_dir))
    mask_list = [os.path.join(mask_dir,m) for m in mask_folder]

    assert len(video_list) == len(mask_list)


    return video_list, mask_list

def read_masks(m_path):
    masks = []
    m_names = os.listdir(m_path)
    m_names.sort()
    for m in m_names:
        m = Image.open(os.path.join(m_path, m))
        m = m.resize((w, h), Image.NEAREST)
        m = np.array(m.convert('L'))
        m = np.array(m > 0).astype(np.uint8)
        m = cv2.dilate(m, cv2.getStructuringElement(
            cv2.MORPH_CROSS, (3, 3)), iterations=4)
        masks.append(Image.fromarray(m*255))
    return masks

def read_frames(f_path):
    frames = []
    f_names = os.listdir(f_path)
    f_names.sort()
    for f in f_names:
        image = cv2.imread(os.path.join(f_path,f))
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        frames.append(image.resize((w,h)))
    return frames


def main_worker():
    #Model Online
    model_name = "FuseFormer_O"
    #Loading
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = importlib.import_module('model.' + model_name)
    model = net.InpaintGenerator().to(device)
    model_path = args.ckpt
    data = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(data, strict=False)
    print('Loading model from: {}'.format(args.ckpt))
    model.eval()

    frame_list, mask_list = get_videos_list(args)
    video_num = len(frame_list)

    generation_time_all = 0
    video_length_all = 0

    if args.evaluate:
        ssim_all = 0
        psnr_all = 0
        output_i3d_activations = []
        real_i3d_activations = []

        # For mean PSNR/SSIM by frame
        count_frames = [0]*200 # Change 200 with maximum video length
        metrics_frames = [[0,0] for i in range(200)]

    ## Inpainting Loop

    for video_no in range(video_num):
        torch.cuda.empty_cache()
        video_name = frame_list[video_no].split("/")[-1]
        print("[Processing: {} - {}]".format(video_name,video_no))

        frames_PIL = read_frames(frame_list[video_no])
        video_length = len(frames_PIL)
        imgs = _to_tensors(frames_PIL).unsqueeze(0)*2-1
        frames = [np.array(f).astype(np.uint8) for f in frames_PIL]

        masks = read_masks(mask_list[video_no])
        binary_masks = [np.expand_dims((np.array(m) != 0).astype(np.uint8), 2) for m in masks]
        masks = _to_tensors(masks).unsqueeze(0)

        imgs, masks = imgs.to(device), masks.to(device)
        comp_frames = [None]*video_length

        video_length_all += (video_length)

        #Timer for FPS measure
        start = time.perf_counter()
        #In Online inpainting, frames are inpainted one by one
        for f in range(0, video_length):
            #The change here is that no frames past f can be used (online model)
            neighbor_ids = [i for i in range(max(0, f-num_neighbors), f)]
            ref_ids = get_ref_index(f, neighbor_ids, ref_step, num_refs)

            with torch.no_grad():
                selected_imgs = imgs[:1, [f]+neighbor_ids+ref_ids, :, :, :]
                selected_masks = masks[:1, [f]+neighbor_ids+ref_ids, :, :, :]
                masked_imgs = selected_imgs*(1-selected_masks)

                pred_img = model(masked_imgs)

                pred_img = (pred_img + 1) / 2
                pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy()*255
                img = np.array(pred_img[0]).astype(np.uint8)*binary_masks[f] + frames[f] * (1-binary_masks[f])
                comp_frames[f] = img


        #End of the timer
        end = time.perf_counter()
        inpaint_time = end-start
        generation_time_all += inpaint_time
        print("Generating {} frames in {} s ({} FPS)".format(video_length, round(inpaint_time,2), round(video_length/inpaint_time,1)))
        print()

        ## Saving inpaintings
        if args.save_results:
            folder = os.path.join(args.save_folder,'FuseFormer_O',video_name)
            if not os.path.isdir(folder):
                os.makedirs(folder)
            for f in range(video_length):
                frame = comp_frames[f]
                frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
                cv2.imwrite(os.path.join(folder, str(f).zfill(5) + ".jpg"), frame)

        ## Evaluating inpaintings
        if args.evaluate:
            comp_PIL = []

            for f in range(video_length):
                count_frames[f] = count_frames[f] + 1
                comp = comp_frames[f]
                comp = cv2.cvtColor(np.array(comp), cv2.COLOR_BGR2RGB)

                # PSNR & SSIM computation (for each frame)
                gt = cv2.cvtColor(np.array(frames[f]).astype(np.uint8), cv2.COLOR_BGR2RGB)
                ssim = metrics.structural_similarity(comp, gt, data_range=255, channel_axis=2, win_size=65)
                psnr = metrics.peak_signal_noise_ratio(gt, comp, data_range=255)

                psnr_all += psnr
                ssim_all += ssim
                metrics_frames[f][0] += psnr
                metrics_frames[f][1] += ssim

                comp_img = Image.fromarray(cv2.cvtColor(comp, cv2.COLOR_BGR2RGB))
                comp_PIL.append(comp_img)

            # FVID computation (for whole video)
            imgs = _to_tensors(comp_PIL).unsqueeze(0).to(device)
            gts = _to_tensors(frames_PIL).unsqueeze(0).to(device)
            output_i3d_activations.append(get_i3d_activations(imgs).cpu().numpy().flatten())
            real_i3d_activations.append(get_i3d_activations(gts).cpu().numpy().flatten())

    print("Finished generating all {} videos".format(video_num))
    print("Mean framerate = {} FPS".format(round(video_length_all/generation_time_all,1)))
    print()

    if args.evaluate:
        fid_score = get_fid_score(real_i3d_activations, output_i3d_activations)
        print("[Final scores]")
        print("PSNR = {}".format(psnr_all/video_length_all))
        print("SSIM = {}".format(ssim_all/video_length_all))
        print("FVID = {}".format(fid_score))

        # Use this to calculate mean PSNR/SSIM by frame
        #print(count_frames)
        #print(metrics_frames)


if __name__ == '__main__':
    main_worker()
