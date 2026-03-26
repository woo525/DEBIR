import glob
import os
import tqdm
import numpy as np
import random
import torch
import cv2
import argparse
import pickle
import math
import data.camera_motion_pipeline_jsrim as rgb2raw


pixel_unshuffle = torch.nn.PixelUnshuffle(2)

def simulate_bursts(img_seq, avg_num, out_dir, burst_num, iso, real_flag=False):
    
    # Sample camera pipeline params.
    rgb2cam = rgb2raw.random_ccm()
    cam2rgb = rgb2cam.inverse()
    rgb_gain, red_gain, blue_gain = rgb2raw.random_gains()
    
    with torch.no_grad():
        
        cnt = 0
        frame_idx = 0
        if real_flag:
            h_ = 736
            w_ = 642
        else:
            h_ = 720
            w_ = 1280  
        img_lin = torch.zeros(3, h_, w_).to(torch.float32).cuda()
        sat_mask = torch.zeros(3, h_, w_).to(torch.float32).cuda()
        
        burst_dir = os.path.join(out_dir, "%04d" %(burst_num))
        if not os.path.exists(burst_dir):
            os.mkdir(burst_dir)
        
        for idx in range(len(img_seq) + 1):
            
            # GT
            if cnt == 0 and idx == 0:

                img_ = cv2.cvtColor(cv2.imread(img_seq[idx]), cv2.COLOR_BGR2RGB)
                img_ = img_[:h_,:w_,:]
                img = (torch.tensor(img_.transpose(2, 0, 1)).cuda() / 255.)
                # Inverts gamma compression.
                img = rgb2raw.gamma_expansion(img).clamp(0, 1)
                # Inverts color correction. & inverts white balance and brightening.
                img_lin = rgb2raw.apply_ccm(img, rgb2cam) 
                img_lin = rgb2raw.apply_gains(img_lin, 1/rgb_gain, 1/red_gain, 1/blue_gain, clamp=False)
                # Mosaic. (3ch to 1ch)
                img_mosaic = torch.zeros_like(img_lin)[0, :, :]
                img_mosaic[0::2, 0::2] = img_lin[0, 0::2, 0::2]
                img_mosaic[0::2, 1::2] = img_lin[1, 0::2, 1::2]
                img_mosaic[1::2, 0::2] = img_lin[1, 1::2, 0::2]
                img_mosaic[1::2, 1::2] = img_lin[2, 1::2, 1::2]
                # Pixel-unshuffle. (1ch to 4ch)
                img_mosaic = pixel_unshuffle(img_mosaic.unsqueeze(0))
                # Save.
                img_rggb = img_mosaic.permute(1, 2, 0).cpu().numpy().astype(np.float32)
                np.save(os.path.join(burst_dir, 'gt.npy'), img_rggb)

                img_lin = torch.zeros(3, h_, w_).to(torch.float32).cuda()
                
                # Save meta-info.
                if real_flag:
                    metadata = {'rgb2cam': rgb2cam, 'cam2rgb': cam2rgb, 'rgb_gain': rgb_gain, 'red_gain': red_gain,
                    'blue_gain': blue_gain, 'smoothstep': False, 'gamma': True, 'iso': iso, 
                    'avg_num': avg_num, 'clip_name': img_seq[idx].split('/')[-3], 'frame_name': img_seq[idx].split('/')[-1].split(".")[0]}
                else:
                    metadata = {'rgb2cam': rgb2cam, 'cam2rgb': cam2rgb, 'rgb_gain': rgb_gain, 'red_gain': red_gain,
                    'blue_gain': blue_gain, 'smoothstep': False, 'gamma': True, 'iso': iso, 
                    'avg_num': avg_num, 'clip_name': img_seq[idx].split('/')[-2], 'frame_name': img_seq[idx].split('/')[-1]}
                meta_path = os.path.join(burst_dir, 'metadata.pkl')
                with open(meta_path, 'wb') as pickle_file:
                    pickle.dump(metadata, pickle_file) 
            
            # Bursts
            if cnt == int(avg_num[frame_idx]):
                
                # Averaging.
                img_lin = img_lin / cnt
                sat_mask = sat_mask / cnt
                # Saturation synthesis.
                alpha = np.random.uniform(low=0.25, high=1.75)
                img_lin += (alpha * sat_mask)
                img_lin = torch.clip(img_lin, min=0., max=1.)
                # Inverts color correction. & inverts white balance and brightening.
                img_lin = rgb2raw.apply_ccm(img_lin, rgb2cam) 
                img_lin = rgb2raw.apply_gains(img_lin, 1/rgb_gain, 1/red_gain, 1/blue_gain, clamp=False)
                # Mosaic. (3ch to 1ch)
                img_mosaic = torch.zeros_like(img_lin)[0, :, :]
                img_mosaic[0::2, 0::2] = img_lin[0, 0::2, 0::2]
                img_mosaic[0::2, 1::2] = img_lin[1, 0::2, 1::2]
                img_mosaic[1::2, 0::2] = img_lin[1, 1::2, 0::2]
                img_mosaic[1::2, 1::2] = img_lin[2, 1::2, 1::2]
                # Add noise.
                shot_noise_level, read_noise_level = rgb2raw.random_noise_levels(iso[frame_idx])
                scale_var = read_noise_level + shot_noise_level * img_mosaic # Add noise
                total_noise = torch.normal(mean=0, std=torch.sqrt(scale_var))
                img_mosaic = (img_mosaic + total_noise).clip(0,1)
                # Pixel-unshuffle. (1ch to 4ch)
                img_mosaic = pixel_unshuffle(img_mosaic.unsqueeze(0))
                # Save.
                img_rggb = img_mosaic.permute(1, 2, 0).cpu().numpy().astype(np.float32)
                np.save(os.path.join(burst_dir, 'burst_%01d.npy' %(frame_idx)), img_rggb)
                
                frame_idx += 1

                cnt = 0
                img_lin = torch.zeros(3, h_, w_).to(torch.float32).cuda()
                sat_mask = torch.zeros(3, h_, w_).to(torch.float32).cuda()

            if idx < len(img_seq):
                
                img_ = cv2.cvtColor(cv2.imread(img_seq[idx]), cv2.COLOR_BGR2RGB)
                img_ = img_[:h_,:w_,:]
                img = torch.tensor(img_.transpose(2, 0, 1)).cuda() / 255.
                
                # Inverts gamma compression.
                img = rgb2raw.gamma_expansion(img).clamp(0, 1)

                # Simulate blur.
                img_lin += img
                sat_mask[img >= 1] += 1

                cnt += 1

def realblur_frames(mode):
    file_path = os.path.join(root_path, "RealBlur_J_%s_list.txt" %(mode))
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    lines = [line.strip() for line in lines]
    clip_list = []
    for line in lines:
        if line.split("/")[-3] not in clip_list:
            clip_list.append(line.split("/")[-3])
    clip_list.sort()
    clip_list = clip_list[:91] # for train1

    if mode == "train":
        num_per_clip = 1100 // len(clip_list)
    else: # "test"
        num_per_clip = 550 // len(clip_list)

    frames_list = []
    for clip_name in clip_list: 
        cand_list = glob.glob(os.path.join(root_path, "RealBlur-J_ECC_IMCORR_centroid_itensity_ref", clip_name, "gt", "*.png"))
        cand_list.sort()
        frames_list.extend(cand_list[:num_per_clip])

    return frames_list
            
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    min_exp = 1/240.

    img_num = 4
    n_max = 64
    print("img_num: %01d" %(img_num))

    test_permutations = [tuple(random.randint(8, n_max) for _ in range(img_num)) for _ in range(1200)]
    train_permutations = test_permutations * 4
    train_permutations_static = test_permutations
    
    test_intervals = []
    for combo in test_permutations:
        curr_exp = [x / 1920. for x in combo]
        curr_iso = []
        max_iso = random.randint(102400, 204800)
        for exp in curr_exp:
            curr_iso.append(max_iso * min_exp / exp)
        test_intervals.append((combo, curr_iso))

    train_intervals = []
    for combo in train_permutations:
        curr_exp = [x / 1920. for x in combo]
        curr_iso = []
        max_iso = random.randint(102400, 204800)
        for exp in curr_exp:
            curr_iso.append(max_iso * min_exp / exp)
        train_intervals.append((combo, curr_iso))
    random.shuffle(train_intervals)

    train_intervals_static = []
    for combo in train_permutations_static:
        curr_exp = [x / 1920. for x in combo]
        curr_iso = []
        max_iso = random.randint(102400, 204800)
        for exp in curr_exp:
            curr_iso.append(max_iso * min_exp / exp)
        train_intervals_static.append(([1 for _ in range(img_num)], curr_iso))
    random.shuffle(train_intervals_static)

    '''==================== set paths ===================='''

    out_path = '../resources/STAGE1_dataset'

    ## To make static bursts from RealBlur
    root_path = '../resources/RealBlur/RealBlur'
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    mode_list = ['train'] # Just for train dataset

    for mode in tqdm.tqdm(mode_list):
        itval_idx = 0 ##
        out_dir_clip = os.path.join(out_path, mode, "RealBlur")
        if not os.path.exists(out_dir_clip):
            os.makedirs(out_dir_clip, exist_ok=True)

        frames_list = realblur_frames(mode)

        for i, frame in enumerate(frames_list):
            intervals, curr_iso = train_intervals_static[itval_idx]
            itval_idx += 1
            final_list = [frame for k in range(img_num)]
            simulate_bursts(final_list, intervals, out_dir_clip, i, curr_iso, real_flag=True)

    ## To make dynamic bursts from GOPRO
    root_path = '../resources/GOPRO_x8_DEBIR'
    
    mode_list = ['train','test']
    
    train1_list = ["GOPR0374_11_00", "GOPR0374_11_02", "GOPR0374_11_03", "GOPR0378_13_00", "GOPR0379_11_00", "GOPR0380_11_00", "GOPR0384_11_01",
                   "GOPR0372_07_00", "GOPR0372_07_01", "GOPR0374_11_01", "GOPR0857_11_00"]

    for mode in tqdm.tqdm(mode_list):
        itval_idx = 0
        if mode == "test":
            clip_list = glob.glob(os.path.join(root_path, mode, '*'))
        else: # train
            clip_list = glob.glob(os.path.join(root_path, "train1", '*'))

        for clip_path in tqdm.tqdm(clip_list):
            if not os.path.isdir(clip_path): 
                continue
            if mode == "train" and clip_path.split("/")[-1] not in train1_list:
                continue
            
            clip_name = clip_path.split('/')[-1]
            out_dir_clip = os.path.join(out_path, mode, clip_name)
            if not os.path.exists(out_dir_clip):
                os.makedirs(out_dir_clip, exist_ok=True)

            # irradiance frame paths
            if mode == "test":
                frames_list = glob.glob(os.path.join(root_path, mode, clip_name, '*.png'))
            else: # train
                frames_list = glob.glob(os.path.join(root_path, "train1", clip_name, '*.png'))
            frames_list = sorted(frames_list)
            frames_list = frames_list[1:]
            
            cl = len(frames_list)
            if img_num > 4:
                max_bl = img_num * n_max
            else:
                max_bl = 4 * 64

            if mode == 'train':
                c_num = (1200 * 4) // 11
            if mode == 'test':
                c_num = 1200 // 11
            inter = math.ceil((c_num * max_bl - cl) / (c_num-1))
            
            for i in tqdm.tqdm(range(int(c_num))):
                idx = i * (max_bl - inter)
                if mode == 'train':
                    intervals, curr_iso = train_intervals[itval_idx]
                    itval_idx += 1
                if mode == 'test':
                    intervals, curr_iso = test_intervals[itval_idx]
                    itval_idx += 1

                # intervals
                frames_list_ = frames_list[idx:idx+int(sum(intervals))-7]

                start_index = 0
                final_list = []
                for count in intervals:
                    final_list.extend(frames_list_[start_index:start_index+int(count)-7])
                    start_index += int(count)

                simulate_bursts(final_list, [x - 7 for x in intervals], out_dir_clip, i, curr_iso)


            



