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
import pickle

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

parser = argparse.ArgumentParser(description='DSTT_OMR')
parser.add_argument("--video", type=str, required=True)
parser.add_argument("--mask", type=str, required=True)
parser.add_argument("--ckpt", type=str, default='checkpoints/DSTT.pth')
parser.add_argument("--width", type=int, default=432)
parser.add_argument("--height", type=int, default=240)
parser.add_argument("--ref_step", type=int, default=10)
parser.add_argument("--num_neighbors", type=int, default=3)
parser.add_argument("--num_refs", type=int, default=3)
parser.add_argument("--refine_window", type=int, default=4)
parser.add_argument("--online_window", type=int, default=3)
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

    print("[Loading I3D model for FID score ..]")
    i3d_model_weight = './checkpoints/i3d_rgb_imagenet.pt'
    #if not os.path.exists(i3d_model_weight):
    #    os.mkdir(os.path.dirname(i3d_model_weight))
    #    urllib.request.urlretrieve('http://www.cmlab.csie.ntu.edu.tw/~zhe2325138/i3d_rgb_imagenet.pt', i3d_model_weight)
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

def get_videos_list(args):
    video_dir = args.video
    video_folder = sorted(os.listdir(video_dir))
    video_list = [os.path.join(video_dir,f) for f in video_folder]

    mask_dir = args.mask
    mask_folder = sorted(os.listdir(mask_dir))
    mask_list = [os.path.join(mask_dir,m) for m in mask_folder]

    assert len(video_list) == len(mask_list)

    return video_list, mask_list

def get_ref_index(f, neighbor_ids, ref_step, num_refs,comp=None):
    """
    Returns the last reference frames before the current frames not in its neighborhood
    Also may check if the frames have been completed before
    ref_steps = sampling factor (10 by default)
    num_refs = number of frames to return
    f = current frame
    neighbor_ids = neighborhood of the current frame
    comp = completed frames
    """
    ref_index = []
    i = f-1
    while(i>=0 and len(ref_index)<num_refs):
        if not i in neighbor_ids and not i%ref_step:
            if comp is not None:
                if comp[i]:
                    ref_index.append(i)
            else:
                ref_index.append(i)
        i -= 1
    return ref_index[::-1]

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

#The refinement process runs on a GPU and inpaint already seen frames in blocks (of size refine_window) to get better inpainting information for the online process
def refine_process(masked_images, video_length, refine_model, refine_stack, computed_frames, current):
    k = args.refine_window
    last = 0
    while(current<video_length):
        if current>=(k-1): #Wait for first frames to pass before starting refine process (because we need a minimum number of frames)
            last = int(current)
            if last<video_length: #Continue the refinement only if the online process isn't already finished itself
                ids = [last-j for j in range(k)][::-1] # k last frames available, including the current one
                ids = ids + get_ref_index(last, ids, ref_step, num_refs)

                input = masked_images[:1, ids, :, :, :]

                #Inpaint the block of frames, then store the results in the refine_stack (located on the other GPU) to be used later by the online process
                with torch.no_grad():
                    attn = refine_model(input)
                    attn = attn.view(8,1,-1,720,512).permute(2,0,1,3,4).to("cuda:1") #Switch to other GPU to complete refine_stack
                    for j in range(len(ids)):
                        if computed_frames[ids[j]]:
                            refine_stack[ids[j]] = 0.5 * refine_stack[ids[j]] + 0.5 * attn[j]
                        else:
                            refine_stack[ids[j]] = attn[j]
                            computed_frames[ids[j]] = True


#The online process runs on the other GPU and works in the same way as other online model, except that is also uses inpainting memory from the refinement process as well
def online_process(masked_images, video_length, online_model, refine_stack, computed_frames, current, res):
    k = args.refine_window
    l = args.online_window
    inpainted = []
    #Store the inpainting memory of the online process that can be used as well
    online_stack = torch.Tensor().to(masked_images.device)
    torch.cuda.synchronize()
    start = time.perf_counter()
    while(current<video_length):
        if current<k or torch.any(computed_frames): #After k first frames, wait for the refinement to start as well
            c = int(current)
            input = masked_images[:1, [c], :, :, :]

            #Ids of the frames from the refinement memory to use
            neighbor_ids = [i for i in range(max(0, c-10), c) if computed_frames[i]][-num_neighbors:]
            ref_ids = get_ref_index(c, neighbor_ids, ref_step, num_refs, computed_frames)
            ids = neighbor_ids + ref_ids

            refine_attn = refine_stack[ids]

            with torch.no_grad():
                #Prediction using information from the refine memory and the current online memory
                pred_img, attn = online_model(input, torch.cat((online_stack[-l:],refine_attn)))
                #Memory update
                online_stack = torch.cat((online_stack,attn.unsqueeze(0)))

                pred_img = (pred_img + 1) / 2
                pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy()*255
                img = np.array(pred_img[0]).astype(np.uint8)
                inpainted.append(img)
            current += 1

    torch.cuda.synchronize()
    end = time.perf_counter()
    inpaint_time = end-start
    print("Generating {} frames in {} s ({} FPS)".format(video_length, round(inpaint_time,2), round(video_length/inpaint_time,1)))
    print()
    res.put((inpainted, inpaint_time))

if __name__ == '__main__':
    mp.set_start_method('spawn')

    #Loading Refinement Model (on GPU 0)
    device_0 = torch.device("cuda:0")
    refine_name = "DSTT_R"
    net_ = importlib.import_module('model.' + refine_name)
    refine_model = net_.InpaintGenerator().to(device_0)
    refine_data = torch.load(args.ckpt, map_location=device_0)
    refine_model.load_state_dict(refine_data['netG'])
    refine_model.eval()
    refine_model.share_memory()
    print("Refinement model loaded")

    #Loading Online-Memory Model (on GPU 1)
    device_1 = torch.device("cuda:1")
    model_name = "DSTT_OM"
    net = importlib.import_module('model.' + model_name)
    online_model = net.InpaintGenerator().to(device_1)
    online_data = torch.load(args.ckpt, map_location=device_1)
    online_model.load_state_dict(online_data['netG'])
    online_model.eval()
    online_model.share_memory()
    print("Online model loaded")

    #Empty GPU memory
    del refine_data
    del online_data
    torch.cuda.empty_cache()

    #Loading all videos
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

        video_name = frame_list[video_no].split("/")[-1]
        print("[Processing: {} - {}]".format(video_name,video_no))

        frames_PIL = read_frames(frame_list[video_no])
        video_length = len(frames_PIL)
        imgs = _to_tensors(frames_PIL).unsqueeze(0)*2-1
        frames = [np.array(f).astype(np.uint8) for f in frames_PIL]

        masks = read_masks(mask_list[video_no])
        binary_masks = [np.expand_dims((np.array(m) != 0).astype(np.uint8), 2) for m in masks]
        masks = _to_tensors(masks).unsqueeze(0)

        video_length_all += (video_length)

        ## Initialisation of the input variables (stored on one of the 2 GPUs)
        #To maximize the speed, we put as much elements as possible on the same GPU as the online process (to access everything quickly without transfer)

        imgs, masks = imgs.to(device_1), masks.to(device_1)
        masked = imgs*(1-masks)

        masked_refine = torch.clone(masked).to(device_0)

        refine_stack = torch.zeros((video_length,8,1,720,512)).to(device_1)
        refine_stack.share_memory_() #Used by both processes so must be shared

        current_frame = torch.tensor([0], device = device_1)
        current_frame.long()
        current_frame.share_memory_() #Also shared

        comp_frames = torch.zeros(video_length, device = device_1).type(torch.bool)
        comp_frames.share_memory_() #Also shared

        result = mp.Queue()

        ### MULTI PROCESSES
        # Both processes are launched jointly, each one running on its own GPU. Some variables are shared (e.g updated on one GPU and read by the other)

        refine_inpainting = mp.Process(target = refine_process, args = (masked_refine, video_length, refine_model, refine_stack, comp_frames, current_frame))

        online_inpainting = mp.Process(target = online_process, args = (masked, video_length, online_model, refine_stack, comp_frames, current_frame, result))

        refine_inpainting.start()
        online_inpainting.start()

        inpainted, dur = result.get()

        online_inpainting.join()
        refine_inpainting.join()

        #Manually cleaning GPU memory as it may not be done properly otherwise (resulting in CUDA OOM)
        del refine_stack
        torch.cuda.empty_cache()


        ## OUTPUTS

        for f in range(video_length):
            inpainted[f] = inpainted[f] * binary_masks[f] + frames[f] * (1-binary_masks[f])
        generation_time_all += dur

        if args.save_results:
            folder = os.path.join(args.save_folder,'DSTT_OMR',video_name)
            if not os.path.isdir(folder):
                os.makedirs(folder)
            for f in range(video_length):
                frame = inpainted[f]
                frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
                cv2.imwrite(os.path.join(folder, str(f).zfill(5) + ".jpg"), frame)

        if args.evaluate:
            comp_PIL = []

            for f in range(video_length):
                count_frames[f] = count_frames[f] + 1
                comp = inpainted[f]
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
            imgs = _to_tensors(comp_PIL).unsqueeze(0).to(device_0)
            gts = _to_tensors(frames_PIL).unsqueeze(0).to(device_0)
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
