import os
import cv2
import torch
import argparse
import numpy as np
from pytorch_lightning import Trainer, seed_everything
from torch.utils.data.dataloader import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.profiler import SimpleProfiler, AdvancedProfiler

seed_everything(50)

from Network_prepare_stqage2w import Burstormer
from datasets.dataset_prepare_stage2w import GoProRAW2RAW

class Args:
    def __init__(self):
        self.exp_name = "NUNI_US64"
        self.patch_size = 256
        self.pred_exp = 64
        self.NUM_WORKERS = 6
        
args = Args()

ckpt_path = "logs/checkpoints/%s.ckpt" %(args.exp_name)
print(ckpt_path)

model = Burstormer()
model = Burstormer.load_from_checkpoint(ckpt_path)
model.cuda()
model.summarize()

trainer = Trainer(gpus=-1,
                auto_select_gpus=True,
                accelerator='ddp',
                max_epochs=300,
                precision=16,
                gradient_clip_val=0.01,
                benchmark=True,
                deterministic=False,
                val_check_interval=0.125,                   
                progress_bar_refresh_rate=1,
                enable_progress_bar=True,
                logger=False)

t_flag = ["train","test"]

lst = [[8,8,8,8],[16,16,16,16],[24,24,24,24],[32,32,32,32],[8,16,24,32]]

for flag in t_flag:
    for exp_list in lst:
        if flag == "train":
            train_dataset = GoProRAW2RAW(root="../resources/GOPRO_x8_DEBIR", split='train', patch_size=args.patch_size, args=args, exp_list=exp_list)
            t_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=args.NUM_WORKERS, pin_memory=True)
        else: # "test"
            test_dataset = GoProRAW2RAW(root="../resources/GOPRO_x8_DEBIR", split='test', patch_size=512, args=args, exp_list=exp_list)
            t_loader = DataLoader(test_dataset, batch_size=1, num_workers=args.NUM_WORKERS, pin_memory=True)
        
        psnrs = trainer.predict(model, t_loader)
        
        result_dict = {key: value for key, value in psnrs}

        file_path = "./warm_up_labels/%s" %(args.exp_name)
        if not os.path.exists(file_path):
            os.makedirs(file_path, exist_ok=True) 
        
        if exp_list[0] == exp_list[1]:
            with open(os.path.join(file_path,"psnrs_%02d_%s.txt") %(exp_list[0],flag), "w") as file:
                for key, value in result_dict.items():
                    file.write(f"{key}: {value}\n")
        else:
            with open(os.path.join(file_path,"psnrs_P%01d_%s.txt") %(exp_list[1]-exp_list[0],flag), "w") as file:
                for key, value in result_dict.items():
                    file.write(f"{key}: {value}\n")
