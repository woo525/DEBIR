import torch
import os
import cv2
import random
import numpy as np
import glob
import pickle
import torch.nn.functional as F
import torch.nn as nn


class GoProRAW2RAW(torch.utils.data.Dataset):
    def __init__(self, root, split='train', patch_size=256, args=None, exp_list=[]):
        super().__init__()
        
        self.exp_list = exp_list

        self.args = args
        self.patch_size = patch_size
        self.pred_exp_time = args.pred_exp - 7 if args is not None else 9
        self.total_exp_time = self.pred_exp_time + 25 * 4 + 1 + 7 * 3 
        self.total_exp_time += 57 - self.pred_exp_time # 179
        
        self.min_exp = 1/240.

        self.burst_list = {}
        # gopro dataset
        video_names_ = [name for name in os.listdir(os.path.join(root, split)) if os.path.isdir(os.path.join(root, split))]
        video_names = sorted(video_names_)
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
                        for hi in range(1): 
                            for wi in range(2):
                                self.irr_index.append((name, idx, hi, wi))
                else: # realblur dataset
                    k_list = [k for k in range(len(self.burst_list[name]))]
                    for idx in k_list:
                        for hi in range(1): 
                            for wi in range(1):
                                self.irr_index.append((name, idx, hi, wi))
            ## test ##
            else:
                if name != "realblur": # gopro dataset
                    k_list = [k for k in range(len(self.burst_list[name])//self.total_exp_time-1)]
                    for idx in k_list:
                        self.irr_index.append((name, idx, 0, 0))
                else: # realblur dataset
                    k_list = [k for k in range(len(self.burst_list[name]))]
                    for idx in k_list:
                        self.irr_index.append((name, idx, 0, 0))
        
        # load pre-generated meta-info
        file_name = "meta_info.pkl"
        print(">>>>>>>>>>"+file_name+"<<<<<<<<<<")
        with open(file_name, "rb") as file:
            loaded_data = pickle.load(file)
        self.meta_info = loaded_data[split][:len(self.irr_index)]

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
            for img_path in self.burst_list[name][k*self.total_exp_time+(57-self.pred_exp_time):(k+1)*self.total_exp_time-(128-sum(self.exp_list))]:
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
            img_stacked = np.stack([img] * (self.total_exp_time-(128-sum(self.exp_list))-(57-self.pred_exp_time)), axis=0)
            irr_seq = torch.from_numpy(img_stacked).permute(0, 3, 1, 2).float() / 255.0
        
        # meta_info
        rgb2cam = self.meta_info[index]["rgb2cam"]
        cam2rgb = self.meta_info[index]["cam2rgb"]
        rgb_gain = self.meta_info[index]["rgb_gain"]
        red_gain = self.meta_info[index]["red_gain"]
        blue_gain = self.meta_info[index]["blue_gain"]  
        max_iso = self.meta_info[index]["max_iso"]  
        pred_iso = max_iso / 102400. - 1
        meta_info = {'rgb2cam': rgb2cam, 'cam2rgb': cam2rgb, 'rgb_gain': rgb_gain, 'red_gain': red_gain, 'blue_gain': blue_gain, 'max_iso': max_iso,
                     'exp_name': self.args.exp_name.split('/')[-1], 'clip_name': name, 'pred_iso': pred_iso, 'k':k, 'hi':hi, 'wi':wi}
        
        # gt
        irr = irr_seq[self.pred_exp_time].detach().clone()
        gt = torch.zeros(1, 4, irr.shape[1]//2, irr.shape[2]//2)
        irr = self.gamma_reverse(irr).clamp(0,1) 
        irr = self.apply_ccm(irr, rgb2cam) 
        irr = self.apply_gains(irr, 1/rgb_gain, 1/red_gain, 1/blue_gain, clamp=False) 
        irr = self.mosaic(irr.unsqueeze(0))
        gt = self.pixel_unshuffle(irr).squeeze() 
        
        # bursts
        total_irrs = irr_seq[self.pred_exp_time:].detach().clone()
        
        exp_length = 4

        if self.split == "train":
            bursts = torch.empty(exp_length, 4, self.patch_size//2, self.patch_size//2)
        else: # "test"
            bursts = torch.empty(exp_length, 4, 256, 256)

        in_exp_time = [x - 7 for x in self.exp_list]
        if index == 0:
            print(in_exp_time) 
        
        s_idx = 0
        iso_list = []
        for i in range(len(in_exp_time)):
            curr_irrs = total_irrs[s_idx:s_idx+in_exp_time[i]].detach().clone()
            curr_irrs = self.gamma_reverse(curr_irrs) 
            curr_irrs = torch.sum(curr_irrs, axis=0) / in_exp_time[i]
            curr_irrs = self.apply_ccm(curr_irrs, rgb2cam) 
            curr_irrs = self.apply_gains(curr_irrs, 1/rgb_gain, 1/red_gain, 1/blue_gain, clamp=False) 
            curr_irrs = self.mosaic(curr_irrs.unsqueeze(0)) 
            curr_exp_time = 1/1920. * (in_exp_time[i] + 7.)
            curr_iso = torch.tensor(max_iso * self.min_exp / curr_exp_time)
            
            iso_list.append(max_iso * self.min_exp / curr_exp_time)
            
            shot_noise, read_noise = self.random_noise_levels(curr_iso)
            scale_var = read_noise + shot_noise * curr_irrs
            total_noise = torch.normal(mean=0, std=torch.sqrt(scale_var))
            pred_img = (curr_irrs + total_noise).clip(0,1)
            pred_img = self.pixel_unshuffle(pred_img).squeeze()

            bursts[i] = pred_img.detach().clone()
            s_idx += in_exp_time[i] + 7
        
        meta_info['iso'] = iso_list
        flow_vectors = []
        
        return bursts, gt, flow_vectors, meta_info

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
        else:
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
    
    def gamma_reverse(self, pre_img): 
        Mask = lambda x: (x>0.04045).float()
        sRGBLinearize = lambda x,m: m * ((m * x + 0.055) / 1.055) ** 2.4 + (1-m) * (x / 12.92)
        return  sRGBLinearize(pre_img, Mask(pre_img))

    def apply_ccm(self, image, ccm):
        """Applies a color correction matrix."""
        assert image.dim() == 3 and image.shape[0] == 3
        
        shape = image.shape
        # image = image.view(3, -1)
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
        
        bayer_pattern = torch.tensor([[[[0,1],[1,2]]]]).to(x.device)
        
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

