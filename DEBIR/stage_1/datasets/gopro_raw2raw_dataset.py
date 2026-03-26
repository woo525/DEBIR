import torch
import os
import cv2
import random
import numpy as np
import glob


class GoProRAW2RAW(torch.utils.data.Dataset):
    def __init__(self, root, split='train'):
        super().__init__()

        if split in ['train', 'test']:
            self.burst_list = glob.glob(os.path.join(root, split, "*", "*"))
            if split == 'test':
                self.burst_list.sort()
        else:
            raise Exception('Unknown split {}'.format(split))

        self.split = split
        self.root = root

    def get_burst(self, burst_id):
        
        # apply random augmentation
        if self.split=='train':
            flag_aug = random.randint(1,7)
        else:
            flag_aug = 0
        
        # bursts
        burst_size = 4
        for i in range(burst_size):
            img_ = self.data_augmentation(np.load(os.path.join(self.burst_list[burst_id], "burst_%01d.npy" %(i))), flag_aug)
            img = torch.tensor(img_).permute(2, 0, 1).unsqueeze(0)
            
            if i == 0:
                bursts = img
            else:
                bursts = torch.cat((bursts, img), dim=0)
    
        # gt
        gt_ = self.data_augmentation(np.load(os.path.join(self.burst_list[burst_id], "gt.npy")), flag_aug)
        gt = torch.tensor(gt_).permute(2, 0, 1)
        
        # crop
        if self.split == "train": # random crop
            p_size = 128
            hi = random.randint(0, gt.shape[1]-p_size-1)
            wi = random.randint(0, gt.shape[2]-p_size-1)
        if self.split == "test": # center crop
            p_size = 256
            hi = gt.shape[1] // 2 - p_size // 2
            wi = gt.shape[2] // 2 - p_size // 2

        bursts = bursts[:, :, hi:hi+p_size, wi:wi+p_size]
        gt = gt[:, hi:hi+p_size, wi:wi+p_size]
        
        return bursts, gt

    def __len__(self):
        return len(self.burst_list)

    def __getitem__(self, index):
        burst, frame_gt = self.get_burst(index)
        return burst, frame_gt

    def data_augmentation(self, image, mode):
        """
        Performs data augmentation of the input image
        Input:
            image: a cv2 (OpenCV) image
            mode: int. Choice of transformation to apply to the image
                    0 - no transformation
                    1 - flip up and down
                    2 - rotate counterwise 90 degree
                    3 - rotate 90 degree and flip up and down
                    4 - rotate 180 degree
                    5 - rotate 180 degree and flip
                    6 - rotate 270 degree
                    7 - rotate 270 degree and flip
        """
        if mode == 0:
            # original
            out = image
        elif mode == 1:
            # flip up and down
            out = np.flipud(image)
        elif mode == 2:
            # rotate counterwise 90 degree
            out = np.rot90(image)
        elif mode == 3:
            # rotate 90 degree and flip up and down
            out = np.rot90(image)
            out = np.flipud(out)
        elif mode == 4:
            # rotate 180 degree
            out = np.rot90(image, k=2)
        elif mode == 5:
            # rotate 180 degree and flip
            out = np.rot90(image, k=2)
            out = np.flipud(out)
        elif mode == 6:
            # rotate 270 degree
            out = np.rot90(image, k=3)
        elif mode == 7:
            # rotate 270 degree and flip
            out = np.rot90(image, k=3)
            out = np.flipud(out)
        else:
            raise Exception('Invalid choice of image transformation')
        return out.copy()
