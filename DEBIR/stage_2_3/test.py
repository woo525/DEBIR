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
from collections import defaultdict

seed_everything(50)

from Network_DEBIR import Burstormer
from datasets.dataset_stage2m_and_test import GoProRAW2RAW


class Args:
    def __init__(self):
        self.image_dir = "../resources/GOPRO_x8_DEBIR"
        self.exp_name = "debir_obf"
        self.learning_rate = 1e-7 # dummy
        self.eta_min = 1e-10 # dummy
        self.num_epochs = 300 # dummy
        self.patch_size = 256
        self.pred_exp = 64
        self.NUM_WORKERS = 6

        self.visualize = False

args = Args()

ckpts = os.listdir("logs/Track_1/%s" %(args.exp_name))
ckpt_path = os.path.join("logs/Track_1/%s" %(args.exp_name),[file for file in ckpts if "epoch" in file][0])
print(ckpt_path)
ckpt = torch.load(ckpt_path)

new_state_dict = {}
for k, v in ckpt['state_dict'].items():
    new_k = k.replace("prednet", "baenet")
    new_state_dict[new_k] = v

model = Burstormer(args=args)
model.load_state_dict(new_state_dict)
# model = Burstormer.load_from_checkpoint(ckpt_path)

model.cuda()
model.summarize()

trainer = Trainer(gpus=-1, # dummys
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

img_size_lst = ["full"]
for img_size in img_size_lst:

    if img_size =="512":
        test_dataset = GoProRAW2RAW(root=args.image_dir, split='test', patch_size=512, args=args, predict_flag=False)
        test_loader = DataLoader(test_dataset, batch_size=1, num_workers=args.NUM_WORKERS, pin_memory=True)
    else:
        test_dataset = GoProRAW2RAW(root=args.image_dir, split='test', patch_size=512, args=args, predict_flag=True)
        test_loader = DataLoader(test_dataset, batch_size=1, num_workers=args.NUM_WORKERS, pin_memory=True)

    result = trainer.predict(model, test_loader)
    

    ## psnr
    psnr_dict = {key: value[0] for key, value in result}
    sum_gopro = []
    for key, value in psnr_dict.items():
        if "realblur" not in key:
            sum_gopro.append(value)
    gopro_avg = sum(sum_gopro) / len(sum_gopro)
    psnr_dict["GoPro Average"] = gopro_avg
    print(gopro_avg)

    ## ssim
    ssim_dict = {key: value[2] for key, value in result} 
    sum_gopro = []
    for key, value in ssim_dict.items():
        if "realblur" not in key:
            sum_gopro.append(value)
    gopro_avg = sum(sum_gopro) / len(sum_gopro)
    ssim_dict["GoPro Average"] = gopro_avg
    print(gopro_avg)

    ## lpips
    lpips_dict = {key: value[3] for key, value in result}
    sum_gopro = []
    for key, value in lpips_dict.items():
        if "realblur" not in key:
            sum_gopro.append(value)
    gopro_avg = sum(sum_gopro) / len(sum_gopro)
    lpips_dict["GoPro Average"] = gopro_avg
    print(gopro_avg)

    ## save files
    file_path = "./results/%s/full" %(args.exp_name)
    if not os.path.exists(file_path):
        os.makedirs(file_path, exist_ok=True) 
        
    with open(os.path.join(file_path,"psnrs.txt"), "w") as file:
        for key, value in psnr_dict.items():
            file.write(f"{key}: {value}\n")

    with open(os.path.join(file_path,"ssim.txt"), "w") as file:
        for key, value in ssim_dict.items():
            file.write(f"{key}: {value}\n")

    with open(os.path.join(file_path,"lpips.txt"), "w") as file:
        for key, value in lpips_dict.items():
            file.write(f"{key}: {value}\n")

