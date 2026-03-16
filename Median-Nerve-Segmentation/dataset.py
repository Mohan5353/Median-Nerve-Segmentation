#!/usr/bin/env python
# coding: utf-8

import glob
import os
import re
import torch
import random
from torch.utils.data import random_split, DataLoader
import torchvision.transforms as T
from utils.ImagePathDataset import *
from models.VisTR.util import misc as utils


pat=re.compile("(\d+)\D*$")

def key_func(x):
    mat=pat.search(os.path.split(x)[-1]) # match last group of digits
    if mat is None:
        return x
    return "{:>10}".format(mat.group(1)) # right align to 10 digits


def get_dataset(args):
    
    # 1. Get all patient directories and split them patient-wise
    all_patient_dirs = sorted([d for d in glob.glob(args.data_path + '*') if os.path.isdir(d)])
    
    # Shuffle patients with a fixed seed for reproducibility
    random.seed(0)
    random.shuffle(all_patient_dirs)
    
    num_patients = len(all_patient_dirs)
    
    if args.same_dataset:
        train_patients = all_patient_dirs
        val_patients = all_patient_dirs
        test_patients = all_patient_dirs
        print(f"Total patients found: {num_patients}")
        print("Using the same dataset for train, val, and test.")
    else:
        # Default to 80% train, 10% val, 10% test
        n_val = max(1, int(num_patients * 0.1))
        n_test = max(1, int(num_patients * 0.1))
        n_train = num_patients - n_val - n_test
        
        train_patients = all_patient_dirs[:n_train]
        val_patients = all_patient_dirs[n_train:n_train+n_val]
        test_patients = all_patient_dirs[n_train+n_val:]
        
        print(f"Total patients found: {num_patients}")
        print(f"Patient-wise Split: {len(train_patients)} train, {len(val_patients)} val, {len(test_patients)} test")

    def get_images_dirs(patient_list):
        dirs = []
        for p in patient_list:
            found_dirs = glob.glob(os.path.join(p, '*_IMAGES'))
            if found_dirs:
                dirs.extend(found_dirs)
            else:
                # Fallback if the folder itself contains images/masks directly
                if os.path.exists(os.path.join(p, 'images')):
                    dirs.append(p)
        return dirs

    # Determine which set of patients to use
    if args.test_batch_size:
        target_images_dirs = get_images_dirs(test_patients)
    else:
        # For training, we need both train and val dirs
        train_images_dirs = get_images_dirs(train_patients)
        val_images_dirs = get_images_dirs(val_patients)

    # 3. Create data loaders
    if args.test_batch_size:
        test_kwargs = {'batch_size': args.test_batch_size}
    else:
        train_kwargs = {'batch_size': args.batch_size}
        val_kwargs = {'batch_size': args.val_batch_size}

    use_cuda = torch.cuda.is_available()
    
    # Define any image preprocessing steps
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.2316], [0.2038]), #mean #standard deviation
    ])
        
    if use_cuda:
        cuda_kwargs = {'num_workers': args.num_workers,
                       'pin_memory': True}
        
        if args.test_batch_size:
            test_kwargs.update(cuda_kwargs)
        else:
            train_kwargs.update(cuda_kwargs)
            val_kwargs.update(cuda_kwargs)

    if args.model_name in ['unet', 'unetpp', 'attn_unet', 'trans_unet']:
        
        def collect_files(dirs):
            img_list, msk_list = [], []
            for path in dirs:
                frames = sorted(glob.glob(os.path.join(path, 'images/*.jpg')), key=key_func)
                masks = sorted(glob.glob(os.path.join(path, 'masks/*.tif')), key=key_func)
                if not masks: masks = sorted(glob.glob(os.path.join(path, 'masks/*.png')), key=key_func)
                mask_indices = {os.path.splitext(os.path.basename(m))[0]: m for m in masks}
                clip_len = 40 if args.wrist else len(frames)
                for i in range(min(clip_len, len(frames))):
                    frame_idx = os.path.splitext(os.path.basename(frames[i]))[0]
                    if frame_idx in mask_indices:
                        img_list.append(frames[i])
                        msk_list.append(mask_indices[frame_idx])
            return img_list, msk_list

        if args.test_batch_size:
            image_list, mask_list = collect_files(target_images_dirs)
            test_dataset = ImagePathDataset(image_list, mask_list, transform=transform)
            args.n_test = len(test_dataset)
        else:
            train_img, train_msk = collect_files(train_images_dirs)
            val_img, val_msk = collect_files(val_images_dirs)
            train_dataset = ImagePathDataset(train_img, train_msk, transform=transform, aug=args.no_aug)
            val_dataset = ImagePathDataset(val_img, val_msk, transform=transform)
            args.n_train = len(train_dataset)
            args.n_val = len(val_dataset)

    elif args.model_name == 'siam_unet':
        
        def collect_files_siam(dirs):
            curr_img, prev_img, curr_msk, prev_msk = [], [], [], []
            for path in dirs:
                frames = sorted(glob.glob(os.path.join(path, 'images/*.jpg')), key=key_func)
                masks = sorted(glob.glob(os.path.join(path, 'masks/*.tif')), key=key_func)
                if not masks: masks = sorted(glob.glob(os.path.join(path, 'masks/*.png')), key=key_func)
                mask_indices = {os.path.splitext(os.path.basename(m))[0]: m for m in masks}
                paired_frames, paired_masks = [], []
                for i in range(len(frames)):
                    f_idx = os.path.splitext(os.path.basename(frames[i]))[0]
                    if f_idx in mask_indices:
                        paired_frames.append(frames[i])
                        paired_masks.append(mask_indices[f_idx])
                clip_len = 40 if args.wrist else len(paired_frames)
                for i in range(1, min(clip_len, len(paired_frames))):
                    prev_img.append(paired_frames[i-1]); curr_img.append(paired_frames[i])
                    prev_msk.append(paired_masks[i-1]); curr_msk.append(paired_masks[i])
            return curr_img, prev_img, curr_msk, prev_msk

        if args.test_batch_size:
            c_img, p_img, c_msk, p_msk = collect_files_siam(target_images_dirs)
            test_dataset = ImagePathDataset_siam(c_img, p_img, c_msk, p_msk, transform=transform)
            args.n_test = len(test_dataset)
        else:
            tc_img, tp_img, tc_msk, tp_msk = collect_files_siam(train_images_dirs)
            vc_img, vp_img, vc_msk, vp_msk = collect_files_siam(val_images_dirs)
            train_dataset = ImagePathDataset_siam(tc_img, tp_img, tc_msk, tp_msk, transform=transform, aug=args.no_aug)
            val_dataset = ImagePathDataset_siam(vc_img, vp_img, vc_msk, vp_msk, transform=transform)
            args.n_train = len(train_dataset); args.n_val = len(val_dataset)
        
    elif args.model_name == 'lstm_unet':
        
        def collect_files_lstm(dirs):
            img_list, msk_list = [], []
            for path in dirs:
                frames = sorted(glob.glob(os.path.join(path, 'images/*.jpg')), key=key_func)
                masks = sorted(glob.glob(os.path.join(path, 'masks/*.tif')), key=key_func)
                if not masks: masks = sorted(glob.glob(os.path.join(path, 'masks/*.png')), key=key_func)
                mask_indices = {os.path.splitext(os.path.basename(m))[0]: m for m in masks}
                paired_frames, paired_masks = [], []
                for i in range(len(frames)):
                    f_idx = os.path.splitext(os.path.basename(frames[i]))[0]
                    if f_idx in mask_indices:
                        paired_frames.append(frames[i]); paired_masks.append(mask_indices[f_idx])
                
                clip_len = 40 if args.wrist else len(paired_frames)
                if clip_len >= args.num_frames:
                    for i in range(clip_len - args.num_frames + 1):
                        img_list.append(paired_frames[i:i + args.num_frames])
                        msk_list.append(paired_masks[i + args.num_frames - 1])
            return img_list, msk_list

        if args.test_batch_size:
            image_list, mask_list = collect_files_lstm(target_images_dirs)
            test_dataset = ImagePathDataset_lstm(image_list, mask_list, num_frames=args.num_frames, transform=transform)
            args.n_test = len(test_dataset)
        else:
            train_img, train_msk = collect_files_lstm(train_images_dirs)
            val_img, val_msk = collect_files_lstm(val_images_dirs)
            train_dataset = ImagePathDataset_lstm(train_img, train_msk, num_frames=args.num_frames, transform=transform, aug=args.no_aug)
            val_dataset = ImagePathDataset_lstm(val_img, val_msk, num_frames=args.num_frames, transform=transform)
            args.n_train = len(train_dataset); args.n_val = len(val_dataset)

    elif args.model_name == 'vistr' or 'vgg' in args.model_name:
        
        args.batch_size = 1
        num_frames = args.num_frames
        
        def collect_files_vistr(dirs):
            img_list, msk_list = [], []
            for path in dirs:
                frames = sorted(glob.glob(os.path.join(path, 'images/*.jpg')), key=key_func)
                masks = sorted(glob.glob(os.path.join(path, 'masks/*.tif')), key=key_func)
                if not masks: masks = sorted(glob.glob(os.path.join(path, 'masks/*.png')), key=key_func)
                mask_indices = {os.path.splitext(os.path.basename(m))[0]: m for m in masks}
                paired_frames, paired_masks = [], []
                for i in range(len(frames)):
                    f_idx = os.path.splitext(os.path.basename(frames[i]))[0]
                    if f_idx in mask_indices:
                        paired_frames.append(frames[i]); paired_masks.append(mask_indices[f_idx])
                
                clip_len = 40 if args.wrist else len(paired_frames)
                if clip_len >= num_frames:
                    for i in range(clip_len-num_frames+1):
                        img_list.append(paired_frames[i:i+num_frames])
                        msk_list.append(paired_masks[i:i+num_frames])
            return img_list, msk_list

        if args.test_batch_size:
            image_list, mask_list = collect_files_vistr(target_images_dirs)
            test_dataset = ImagePathDataset_vistr(image_list, mask_list, num_frames, transform=make_transform(image_set='val'))
            args.n_test = len(test_dataset)
        else:
            train_img_list, train_msk_list = collect_files_vistr(train_images_dirs)
            train_dataset = ImagePathDataset_vistr(train_img_list, train_msk_list, num_frames, transform=make_transform(image_set='train'), aug=args.no_aug)
            
            val_image_list, val_mask_list = collect_files_vistr(val_images_dirs)
            val_dataset = ImagePathDataset_vistr(val_image_list, val_mask_list, num_frames, transform=make_transform(image_set='val'))

            sampler_train = torch.utils.data.RandomSampler(train_dataset)
            batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
            data_loader_train = DataLoader(train_dataset, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn, num_workers=args.num_workers)
            
            args.n_train = len(train_dataset)
            args.n_val = len(val_dataset)
            
            sampler_val = torch.utils.data.SequentialSampler(val_dataset)
            batch_sampler_val = torch.utils.data.BatchSampler(sampler_val, args.batch_size, drop_last=False)
            data_loader_val = DataLoader(val_dataset, batch_sampler=batch_sampler_val, collate_fn=utils.collate_fn, num_workers=args.num_workers)

            return data_loader_train, data_loader_val
    
    if args.test_batch_size:
        return DataLoader(test_dataset, shuffle=False, **test_kwargs), None
    else:
        train_loader = DataLoader(train_dataset, shuffle=True, **train_kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, **val_kwargs)
        return train_loader, val_loader
