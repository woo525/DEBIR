import torch
import os
import cv2
import random
import numpy as np
import glob
import pickle

import data.camera_motion_pipeline_jsrim as rgb2raw
import torch.nn.functional as F
import torch.nn as nn


def masks_CFA_Bayer(shape, pattern='RGGB'):
    """
    Returns the *Bayer* CFA red, green and blue masks for given pattern.

    Parameters
    ----------
    shape : array_like
        Dimensions of the *Bayer* CFA.
    pattern : unicode, optional
        **{'RGGB', 'BGGR', 'GRBG', 'GBRG'}**,
        Arrangement of the colour filters on the pixel array.

    Returns
    -------
    tuple
        *Bayer* CFA red, green and blue masks.

    Examples
    --------
    >>> from pprint import pprint
    >>> shape = (3, 3)
    >>> pprint(masks_CFA_Bayer(shape))
    (array([[ True, False,  True],
           [False, False, False],
           [ True, False,  True]], dtype=bool),
     array([[False,  True, False],
           [ True, False,  True],
           [False,  True, False]], dtype=bool),
     array([[False, False, False],
           [False,  True, False],
           [False, False, False]], dtype=bool))
    >>> pprint(masks_CFA_Bayer(shape, 'BGGR'))
    (array([[False, False, False],
           [False,  True, False],
           [False, False, False]], dtype=bool),
     array([[False,  True, False],
           [ True, False,  True],
           [False,  True, False]], dtype=bool),
     array([[ True, False,  True],
           [False, False, False],
           [ True, False,  True]], dtype=bool))
    """

    pattern = pattern.upper()

    channels = dict((channel, np.zeros(shape)) for channel in 'RGB')
    for channel, (y, x) in zip(pattern, [(0, 0), (0, 1), (1, 0), (1, 1)]):
        channels[channel][y::2, x::2] = 1

    return tuple(channels[c].astype(bool) for c in 'RGB')

class Demosaic(nn.Module):
    # based on https://github.com/GuoShi28/CBDNet/blob/master/SomeISP_operator_python/Demosaicing_malvar2004.py

    def __init__(self):
        super(Demosaic, self).__init__()

        GR_GB = np.asarray(
            [[0, 0, -1, 0, 0],
             [0, 0, 2, 0, 0],
             [-1, 2, 4, 2, -1],
             [0, 0, 2, 0, 0],
             [0, 0, -1, 0, 0]]) / 8  # yapf: disable

        # [5,5] => rot90 => [1, 1, 5, 5]
        self.GR_GB_pt = torch.tensor(np.rot90(GR_GB, k=2).copy(), dtype=torch.float32)

        Rg_RB_Bg_BR = np.asarray(
            [[0, 0, 0.5, 0, 0],
             [0, -1, 0, -1, 0],
             [-1, 4, 5, 4, - 1],
             [0, -1, 0, -1, 0],
             [0, 0, 0.5, 0, 0]]) / 8  # yapf: disable
        self.Rg_RB_Bg_BR_pt = torch.tensor(np.rot90(Rg_RB_Bg_BR, k=2).copy(), dtype=torch.float32)

        Rg_BR_Bg_RB = np.transpose(Rg_RB_Bg_BR)
        self.Rg_BR_Bg_RB_pt = torch.tensor(np.rot90(Rg_BR_Bg_RB, k=2).copy(), dtype=torch.float32)

        Rb_BB_Br_RR = np.asarray(
            [[0, 0, -1.5, 0, 0],
             [0, 2, 0, 2, 0],
             [-1.5, 0, 6, 0, -1.5],
             [0, 2, 0, 2, 0],
             [0, 0, -1.5, 0, 0]]) / 8  # yapf: disable

        self.Rb_BB_Br_RR_pt = torch.tensor(np.rot90(Rb_BB_Br_RR, k=2).copy(), dtype=torch.float32)


    def cuda(self, device=None):
        self.GR_GB_pt = self.GR_GB_pt.cuda(device)
        self.Rg_RB_Bg_BR_pt = self.Rg_RB_Bg_BR_pt.cuda(device)
        self.Rg_BR_Bg_RB_pt = self.Rg_BR_Bg_RB_pt.cuda(device)
        self.Rb_BB_Br_RR_pt = self.Rb_BB_Br_RR_pt.cuda(device)


    def forward(self, CFA_inputs, pattern='RGGB'):
        batch_size, c, h, w = CFA_inputs.shape
        R_m, G_m, B_m = masks_CFA_Bayer([h, w], pattern)

        # CFA mask
        R_m_pt = torch.from_numpy(R_m[np.newaxis, np.newaxis, :, :]).to(CFA_inputs.device)
        G_m_pt = torch.from_numpy(G_m[np.newaxis, np.newaxis, :, :]).to(CFA_inputs.device)
        B_m_pt = torch.from_numpy(B_m[np.newaxis, np.newaxis, :, :]).to(CFA_inputs.device)

        R = CFA_inputs * R_m_pt
        G = CFA_inputs * G_m_pt
        B = CFA_inputs * B_m_pt

        # True : GR_GB, False : G
        GR_GB_result = F.conv2d(CFA_inputs, weight=self.GR_GB_pt.expand(1, 1, -1, -1), padding=2, groups=1)
        Rm_Bm = np.logical_or(R_m, B_m)[np.newaxis, np.newaxis, :, :]
        Rm_Bm = np.tile(Rm_Bm, [batch_size, 1, 1, 1])
        Rm_Bm_pt = torch.tensor(Rm_Bm.copy(), dtype=torch.bool).to(CFA_inputs.device)
        G = GR_GB_result * Rm_Bm_pt + G * torch.logical_not(Rm_Bm_pt)

        RBg_RBBR = F.conv2d(CFA_inputs, weight=self.Rg_RB_Bg_BR_pt.expand(1, 1, -1, -1), padding=2, groups=1)
        RBg_BRRB = F.conv2d(CFA_inputs, weight=self.Rg_BR_Bg_RB_pt.expand(1, 1, -1, -1), padding=2, groups=1)
        RBgr_BBRR = F.conv2d(CFA_inputs, weight=self.Rb_BB_Br_RR_pt.expand(1, 1, -1, -1), padding=2, groups=1)

        # Red rows.
        R_r = np.transpose(np.any(R_m == 1, axis=1)[np.newaxis]) * np.ones(R_m.shape)
        # Red columns.
        R_c = np.any(R_m == 1, axis=0)[np.newaxis] * np.ones(R_m.shape)
        # Blue rows.
        B_r = np.transpose(np.any(B_m == 1, axis=1)[np.newaxis]) * np.ones(B_m.shape)
        # Blue columns
        B_c = np.any(B_m == 1, axis=0)[np.newaxis] * np.ones(B_m.shape)

        # rg1g2b
        Rr_Bc = R_r * B_c
        Br_Rc = B_r * R_c

        Rr_Bc = np.tile(Rr_Bc[np.newaxis, np.newaxis, :, :], [batch_size, 1, 1, 1])
        Br_Rc = np.tile(Br_Rc[np.newaxis, np.newaxis, :, :], [batch_size, 1, 1, 1])
        Rr_Bc_pt = torch.tensor(Rr_Bc.copy(), dtype=torch.bool).to(CFA_inputs.device)
        Br_Rc_pt = torch.tensor(Br_Rc.copy(), dtype=torch.bool).to(CFA_inputs.device)

        R = RBg_RBBR * Rr_Bc_pt + R * torch.logical_not(Rr_Bc_pt)
        R = RBg_BRRB * Br_Rc_pt + R * torch.logical_not(Br_Rc_pt)

        Br_Rc = B_r * R_c
        Rr_Bc = R_r * B_c

        Br_Rc = np.tile(Br_Rc[np.newaxis, np.newaxis, :, :], [batch_size, 1, 1, 1])
        Rr_Bc = np.tile(Rr_Bc[np.newaxis, np.newaxis, :, :], [batch_size, 1, 1, 1])
        Br_Rc_pt = torch.tensor(Br_Rc.copy(), dtype=torch.bool).to(CFA_inputs.device)
        Rr_Bc_pt = torch.tensor(Rr_Bc.copy(), dtype=torch.bool).to(CFA_inputs.device)

        B = RBg_RBBR * Br_Rc_pt + B * torch.logical_not(Br_Rc_pt)
        B = RBg_BRRB * Rr_Bc_pt + B * torch.logical_not(Rr_Bc_pt)

        Br_Bc = B_r * B_c
        Rr_Rc = R_r * R_c

        Br_Bc = np.tile(Br_Bc[np.newaxis, np.newaxis, :, :], [batch_size, 1, 1, 1])
        Rr_Rc = np.tile(Rr_Rc[np.newaxis, np.newaxis, :, :], [batch_size, 1, 1, 1])
        Br_Bc_pt = torch.tensor(Br_Bc.copy(), dtype=torch.bool).to(CFA_inputs.device)
        Rr_Rc_pt = torch.tensor(Rr_Rc.copy(), dtype=torch.bool).to(CFA_inputs.device)

        R = RBgr_BBRR * Br_Bc_pt + R * torch.logical_not(Br_Bc_pt)
        B = RBgr_BBRR * Rr_Rc_pt + B * torch.logical_not(Rr_Rc_pt)

        new_out = torch.cat([R, G, B], dim=1)

        return new_out


class GoProRAW2RAW(torch.utils.data.Dataset):
    def __init__(self, root, split='train', patch_size=256, args=None):
        super().__init__()
        
        self.args = args
        self.patch_size = patch_size
        self.pred_exp_time = args.pred_exp - 7 if args is not None else 9
        self.total_exp_time = self.pred_exp_time + 25 * 4 + 1 + 7 * 3 
        self.total_exp_time += 57 - self.pred_exp_time # 179
        
        self.min_exp = 1/240.

        self.burst_list = {}
        # gopro dataset
        video_names_ = [name for name in os.listdir(os.path.join(root, split)) if os.path.isdir(os.path.join(root, split))]
        video_names = sorted(video_names_) # sort
        for name in video_names:
            self.burst_list[name] = glob.glob(os.path.join(root, split, name, "*"))
            self.burst_list[name].sort() 
        # realblur dataset
        self.burst_list["realblur"] = self.realblur_frames(split)
        self.burst_list["realblur"].sort() 

        # make sets
        self.irr_index = []
        for name in (self.burst_list.keys()):
            ## train ##
            if split == "train":
                if name != "realblur": # gopro dataset
                    k_list = [k for k in range(len(self.burst_list[name])//self.total_exp_time-1)]
                    for idx in k_list:
                        for hi in range(1): # 2
                            for wi in range(2): # 4
                                self.irr_index.append((name, idx, hi, wi))
                else: # realblur dataset
                    k_list = [k for k in range(len(self.burst_list[name]))]
                    for idx in k_list:
                        for hi in range(1): # 2
                            for wi in range(1): # 2
                                self.irr_index.append((name, idx, hi, wi))
            ## test ##
            else:
                if name != "realblur": # gopro dataset
                    k_list = [k for k in range(len(self.burst_list[name])//self.total_exp_time-1)]
                    for idx in k_list:
                        self.irr_index.append((name, idx, 0, 0))
        
        # load pre-generated meta-info
        file_name = "meta_info.pkl"
        print(">>>>>>>>>>>>>>>>>>>>"+file_name+"<<<<<<<<<<<<<<<<<<<")
        with open(file_name, "rb") as file:
            loaded_data = pickle.load(file)
        self.meta_info = loaded_data[split][:len(self.irr_index)]
        
        oracle_dict = {}
        if split == "train":
            file_name = "warm_up_labels/%s/%s_train.txt" %(args.burstormer_type,args.burstormer_type)
        else: # "test"
            file_name = "warm_up_labels/%s/%s_test.txt" %(args.burstormer_type,args.burstormer_type)
        with open(file_name, "r") as f:
            for line in f:
                key, value = line.strip().split(": ")
                if value == "P8":
                    oracle_dict[key] = value
                else:
                    oracle_dict[key] = float(value)
        
        for i in range(len(self.meta_info)):
            
            name = self.irr_index[i][0]
            k = str(self.irr_index[i][1])
            hi = str(self.irr_index[i][2])  
            wi = str(self.irr_index[i][3])
            new_key = name +'/'+ k +'/'+ hi +'/'+ wi
            self.meta_info[i]['e_gt'] = oracle_dict[new_key]
            
        self.demosaicking_process = Demosaic()

        self.split = split
        self.root = root
        self.pixel_unshuffle = torch.nn.PixelUnshuffle(2)
    
    def __len__(self):
        return len(self.irr_index)

    def __getitem__(self, index):

        name, k, hi, wi = self.irr_index[index] # video_name / first_frame_idx / patch_hi / patch_wi
        
        # irr_seq
        images = []
        if name != "realblur": # gopro dataset
            for img_path in self.burst_list[name][k*self.total_exp_time+(57-self.pred_exp_time):(k+1)*self.total_exp_time]:
                img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                h, w, c = img.shape
                h_off = h//2 - self.patch_size//2
                w_off = w//2 + (wi - 1) * self.patch_size
                if self.split == "train":
                    images.append(img[h_off:h_off+self.patch_size,w_off:w_off+self.patch_size,:])
                else: # test
                    images.append(img[h//2-256:h//2+256,w//2-256:w//2+256,:])
            img_stacked = np.stack(images, axis=0)
            irr_seq = torch.from_numpy(img_stacked).permute(0, 3, 1, 2).float() / 255.0
        else: # realblur dataset
            img_path = self.burst_list[name][k]
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            h, w, c = img.shape
            h_off = h//2 - self.patch_size//2 
            w_off = w//2 - self.patch_size//2 
            if self.split == "train":
                img = img[h_off:h_off+self.patch_size,w_off:w_off+self.patch_size,:]
            else: # test
                img = img[h//2-256:h//2+256,w//2-256:w//2+256,:]
            img_stacked = np.stack([img] * (self.total_exp_time-(57-self.pred_exp_time)), axis=0)
            irr_seq = torch.from_numpy(img_stacked).permute(0, 3, 1, 2).float() / 255.0
        
        # meta_info
        rgb2cam = self.meta_info[index]["rgb2cam"]
        cam2rgb = self.meta_info[index]["cam2rgb"]
        rgb_gain = self.meta_info[index]["rgb_gain"]
        red_gain = self.meta_info[index]["red_gain"]
        blue_gain = self.meta_info[index]["blue_gain"]  
        max_iso = self.meta_info[index]["max_iso"]  
        pred_iso = max_iso / 102400. - 1 # normalized
        e_gt = 0
        if "e_gt" in self.meta_info[index]:
            e_gt = self.meta_info[index]['e_gt']

        meta_info = {'rgb2cam': rgb2cam, 'cam2rgb': cam2rgb, 'rgb_gain': rgb_gain, 'red_gain': red_gain, 'blue_gain': blue_gain, 'max_iso': max_iso,
                     'exp_name': self.args.exp_name.split('/')[-1], 'clip_name': name, 'pred_iso': pred_iso, 'e_gt': e_gt, 'split': self.split}
        
        # gt
        irr = irr_seq[self.pred_exp_time].detach().clone()
        gt = torch.zeros(1, 4, irr.shape[1]//2, irr.shape[2]//2) # [1, 4, 128, 128]
        irr = self.gamma_reverse(irr).clamp(0,1) # gamma expansion
        irr = self.apply_ccm(irr, rgb2cam) # lin to cam color space
        irr = self.apply_gains(irr, 1/rgb_gain, 1/red_gain, 1/blue_gain, clamp=False) # inverse white-balance
        irr = self.mosaic(irr.unsqueeze(0)) # mosaic (3ch >> 1ch)
        gt = self.pixel_unshuffle(irr).squeeze() # pixel-unshuffle (1ch >> 4ch)
        
        ## pred1
        curr_irrs = irr_seq[:9].detach().clone() # irr_seq >> (b, 1281, c, h, w)
        curr_irrs = self.gamma_reverse(curr_irrs) # gamma expansion
        curr_irrs = torch.sum(curr_irrs, axis=0) / 9. # blur synthesis
        curr_irrs = self.apply_ccm(curr_irrs, rgb2cam) # lin to cam color space
        curr_irrs = self.apply_gains(curr_irrs, 1/rgb_gain, 1/red_gain, 1/blue_gain, clamp=False) # inverse white-balance
        curr_irrs = self.mosaic(curr_irrs.unsqueeze(0)) # mosaic (3ch to 1ch)
        curr_exp_time = 1/1920. * (9. + 7.)
        curr_iso = torch.tensor(max_iso * self.min_exp / curr_exp_time)
        if self.split == "test":
            shot_noise, read_noise = self.random_noise_levels_test(curr_iso)
        else: # "train"
            shot_noise, read_noise = self.random_noise_levels(curr_iso)
        scale_var = read_noise + shot_noise * curr_irrs # add noise (heteroscedastic Gaussian random noise)
        total_noise = torch.normal(mean=0, std=torch.sqrt(scale_var))
        curr_irrs = (curr_irrs + total_noise).clip(0,1)

        curr_irrs = self.demosaicking_process(curr_irrs)[0] # Demosaic
        curr_irrs = self.apply_gains(curr_irrs, rgb_gain, red_gain, blue_gain, clamp=True) # Inverts WB
        curr_irrs = self.apply_ccm(curr_irrs, cam2rgb) # Inverts CC
        curr_irrs = rgb2raw.gamma_compression(curr_irrs).clip(0, 1) # Gamma compression
     
        curr_irrs = curr_irrs.permute(1, 2, 0).numpy()
        h, w, c = curr_irrs.shape
        new_size = (int(w * 0.5), int(h * 0.5))
        curr_irrs = cv2.resize(curr_irrs, new_size, interpolation=cv2.INTER_AREA)
        curr_irrs = torch.tensor(curr_irrs).permute(2, 0, 1)

        pred1 = curr_irrs

        ## pred2
        curr_irrs = irr_seq[self.pred_exp_time-16:self.pred_exp_time-7].detach().clone() # irr_seq >> (b, 1281, c, h, w)
        curr_irrs = self.gamma_reverse(curr_irrs) # gamma expansion
        curr_irrs = torch.sum(curr_irrs, axis=0) / 9. # blur synthesis
        curr_irrs = self.apply_ccm(curr_irrs, rgb2cam) # lin to cam color space
        curr_irrs = self.apply_gains(curr_irrs, 1/rgb_gain, 1/red_gain, 1/blue_gain, clamp=False) # inverse white-balance
        curr_irrs = self.mosaic(curr_irrs.unsqueeze(0)) # mosaic (3ch to 1ch)
        curr_exp_time = 1/1920. * (9. + 7.)
        curr_iso = torch.tensor(max_iso * self.min_exp / curr_exp_time)
        if self.split == "test":
            shot_noise, read_noise = self.random_noise_levels_test(curr_iso)
        else: # "train"
            shot_noise, read_noise = self.random_noise_levels(curr_iso)
        scale_var = read_noise + shot_noise * curr_irrs # add noise (heteroscedastic Gaussian random noise)
        total_noise = torch.normal(mean=0, std=torch.sqrt(scale_var))
        curr_irrs = (curr_irrs + total_noise).clip(0,1)

        curr_irrs = self.demosaicking_process(curr_irrs)[0] # Demosaic
        curr_irrs = self.apply_gains(curr_irrs, rgb_gain, red_gain, blue_gain, clamp=True) # Inverts WB
        curr_irrs = self.apply_ccm(curr_irrs, cam2rgb) # Inverts CC
        pred = curr_irrs.detach().clone().clip(0, 1)
        curr_irrs = rgb2raw.gamma_compression(curr_irrs).clip(0, 1) # Gamma compression

        curr_irrs = curr_irrs.permute(1, 2, 0).numpy()
        h, w, c = curr_irrs.shape
        new_size = (int(w * 0.5), int(h * 0.5))
        curr_irrs = cv2.resize(curr_irrs, new_size, interpolation=cv2.INTER_AREA)
        curr_irrs = torch.tensor(curr_irrs).permute(2, 0, 1)

        pred2 = curr_irrs

        return pred1, pred2, pred, irr_seq[self.pred_exp_time:], gt, meta_info


    def realblur_frames(self, mode):
        file_path = "../resources/RealBlur/RealBlur/RealBlur_J_%s_list.txt" %(mode)
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
        lines = [line.strip() for line in lines]
        clip_list = []
        for line in lines:
            if line.split("/")[-3] not in clip_list:
                clip_list.append(line.split("/")[-3])
        clip_list.sort()
        
        if mode == "train":
            clip_list = clip_list[91:]
            num_per_clip = 1 
        else: # "test"
            num_per_clip = 5

        frames_list = []
        for clip_name in clip_list: 
            cand_list = glob.glob(os.path.join("../resources/RealBlur/RealBlur/RealBlur-J_ECC_IMCORR_centroid_itensity_ref", clip_name, "gt", "*.png"))
            cand_list.sort()
            frames_list.extend(cand_list[:num_per_clip])

        return frames_list
    
    def get_p_offset(self, img, p_size, rand_flag):
        h, w, c = img.shape
        if rand_flag == True:
            h_offset = random.randint(0, h-p_size-1)
            w_offset = random.randint(0, w-p_size-1)
        else: # random == False
            h_offset = h // 2 - p_size // 2
            w_offset = w // 2 - p_size // 2
        
        return h_offset, w_offset

    def random_noise_levels(self, iso):
        """Generates random noise levels from a log-log linear distribution."""
        iso2shot = lambda x: 9.2857e-07 * x + 8.1006e-05
        shot_value = iso2shot(iso)
        shot_noise = shot_value + torch.normal(mean=0.0, std=5e-05, size=shot_value.size())

        while shot_noise <= 0:
            shot_value = iso2shot(iso)
            shot_noise = shot_value + torch.normal(mean=0.0, std=5e-05, size=shot_value.size())

        log_shot_noise = torch.log(shot_noise)
        logshot2logread = lambda x: 2.2282 * x + 0.45982
        logread_value = logshot2logread(log_shot_noise)
        log_read_noise = logread_value + torch.normal(mean=0.0, std=0.25, size=logread_value.size())
        read_noise = torch.exp(log_read_noise)

        while read_noise <= 0:
            logread_value = logshot2logread(log_shot_noise)
            log_read_noise = logread_value + torch.normal(mean=0.0, std=0.25, size=logread_value.size())
            read_noise = torch.exp(log_read_noise)

        return shot_noise, read_noise
    
    def random_noise_levels_test(self, iso):
        """Generates random noise levels from a log-log linear distribution."""
        iso2shot = lambda x: 9.2857e-07 * x + 8.1006e-05
        shot_value = iso2shot(iso)
        shot_noise = shot_value

        while shot_noise <= 0:
            shot_value = iso2shot(iso)
            shot_noise = shot_value

        log_shot_noise = torch.log(shot_noise)
        logshot2logread = lambda x: 2.2282 * x + 0.45982
        logread_value = logshot2logread(log_shot_noise)
        log_read_noise = logread_value
        read_noise = torch.exp(log_read_noise)

        while read_noise <= 0:
            logread_value = logshot2logread(log_shot_noise)
            log_read_noise = logread_value
            read_noise = torch.exp(log_read_noise)

        return shot_noise, read_noise
    
    def gamma_reverse(self, pre_img): 
        Mask = lambda x: (x>0.04045).float()
        sRGBLinearize = lambda x,m: m * ((m * x + 0.055) / 1.055) ** 2.4 + (1-m) * (x / 12.92)
        return  sRGBLinearize(pre_img, Mask(pre_img))

    def apply_ccm(self, image, ccm):
        """Applies a color correction matrix."""
        assert image.dim() == 3 and image.shape[0] == 3
        
        shape = image.shape
        image = image.reshape(3, -1)
        ccm = ccm.to(image.device).type_as(image)
        
        image = torch.mm(ccm, image)

        return image.view(shape)

    def apply_gains(self, image, rgb_gain, red_gain, blue_gain, clamp=True):
        """Inverts gains while safely handling saturated pixels."""
        assert image.dim() == 3 and image.shape[0] in [3, 4]
        
        if image.shape[0] == 3:
            gains = torch.tensor([red_gain, 1.0, blue_gain]).to(image.device) * rgb_gain
        else:
            gains = torch.tensor([red_gain, 1.0, 1.0, blue_gain]).to(image.device) * rgb_gain
        gains = gains.view(-1, 1, 1)
        gains = gains.to(image.device).type_as(image)

        if clamp:
            return (image * gains).clamp(0.0, 1.0)
        else:
            return (image * gains)
    
    def gather(self, x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """ Gathers pixels from `x` according to `index` which follows bayer pattern.
        NOTE: Avoid using `torch.gather` since it does not supports deterministic mode.

        Args:
            x: Tensor of shape :math:`(N, C_{in}, H, W)`.
            index: Gather index with bayer pattern of shape :math:`(1|N, C_{out}, 2, 2)`.

        Returns:
            Gathered values of shape :math:`(N, C_{out}, H, W)`.
        """
        assert x.dim()     == 4, f"`x` must be 4-dimensional, got {x.dim()}"
        assert index.dim() == 4, f"`index` must be 4-dimensional, got {index.dim()}"

        xN, xC, xH, xW = x.shape
        iN, iC, iH, iW = index.shape
        C_in, C_out = xC, iC

        assert iN == xN or iN == 1, \
            f"Batch dimension of `index` must be `1` or equal to `x`, got {iN}"
        assert xH % 2 == 0 and xW % 2 == 0, \
            f"`x` must have even height and width, got {xH}x{xW}"
        assert iH == 2 and iW == 2, \
            f"`index` must be 2x2 size, got {iH}x{iW}"

        # x_down: (N, 4*C_in, H/2, W/2)
        x_down = F.pixel_unshuffle(x, 2)
        # x_quat: (N, 4, 1, C_in, H/2, W/2)
        x_quat = x_down.reshape((xN, -1, 4, xH//2, xW//2)).swapaxes(1, 2).unsqueeze(2)

        # i_down: (1|N, 4*C_out)
        i_down = F.pixel_unshuffle(index, 2).squeeze(-1).squeeze(-1)
        # i_quat: (1|N, 4, C_out, 1)
        i_quat = i_down.reshape((iN, -1, 4)).swapaxes(1, 2).unsqueeze(3)

        # mask_index: (N, 4, C_out, C_in)
        mask_index = torch.arange(C_in, dtype=index.dtype, device=index.device).repeat(xN, 4, C_out, 1)
        # mask: (N, 4, C_out, C_in)
        mask = (mask_index == i_quat)

        # x_gath: (N, 4, C_out, H/2, W/2)
        x_gath = (x_quat * mask.unsqueeze(-1).unsqueeze(-1)).sum(dim=3)

        # y_down: (N, 4*C_out, H/2, W/2)
        y_down = x_gath.swapaxes(1, 2).flatten(start_dim=1, end_dim=2)
        # y: (N, C_out, H, W)
        y = F.pixel_shuffle(y_down, 2)

        return y

    def mosaic(self, x: torch.Tensor) -> torch.Tensor:
        """ Mosaicing RGB images to Bayer pattern. 

        Args:
            x: RGB image of shape :math:`(N, 3, H, W)`.
            bayer_pattern: Bayer pattern of `x` of shape :math:`(1|N, 1, 2, 2)`.

        Returns:
            Mosaicked image of shape :math:`(N, 1, H, W)`.
        """
        
        bayer_pattern = torch.tensor([[[[0,1],[1,2]]]]).to(x.device) ## RGGB
        
        assert x.dim()             == 4, f"`x` must be 4-dimensional, got {x.dim()}"
        assert bayer_pattern.dim() == 4, f"`bayer_pattern` must be 4-dimensional, got {bayer_pattern.dim()}"

        xN, xC, xH, xW = x.shape
        bN, bC, bH, bW = bayer_pattern.shape

        assert xC == 3, f"Channel dimension of `x` must be 3, got {xC}"
        assert xH % 2 == 0 and xW % 2 == 0, f"`x` must have even height and width, got {xH}x{xW}"
        assert bN == xN or bN == 1, f"Batch dimension of `bayer_mask` must be `1` or equal to `x`, got {bN}"
        assert bC == 1, f"Channel dimension of `bayer_mask` must be 1, got {bC}"
        assert bH == 2 and bW == 2, f"Height and width dimension of `bayer_mask` must be 2, got {bH}x{bW}"

        y = self.gather(x, bayer_pattern)
        return y


