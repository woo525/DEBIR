import cv2
import sys
import torch
import numpy as np
import argparse
import glob
import os
import tqdm
import gc
import shutil

'''==========import from our code=========='''
sys.path.append('.')
import config as cfg
from Trainer import Model
from benchmark.utils.padder import InputPadder

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='ours_t', type=str)
parser.add_argument('--n', default=8, type=int)
parser.add_argument('--start_video', default=0, type=int)
parser.add_argument('--end_video', default=-1, type=int)
args = parser.parse_args()
assert args.model in ['ours_t', 'ours_small_t'], 'Model not exists!'

'''==========Model setting=========='''
TTA = True
if args.model == 'ours_small_t':
    TTA = False
    cfg.MODEL_CONFIG['LOGNAME'] = 'ours_small_t'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F = 16,
        depth = [2, 2, 2, 2, 2]
    )
else:
    cfg.MODEL_CONFIG['LOGNAME'] = 'ours_t'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F = 32,
        depth = [2, 2, 2, 4, 4]
    )
model = Model(-1) 
model.load_model()
model.eval()
model.device()

root_path = '../resources/GOPRO' # "GOPRO_Large_all"

out_path = '../resources/GOPRO_x8_DEBIR' 
if not os.path.exists(out_path):
    os.mkdir(out_path)

mode_list = ['train','test']
continue_list = []

for mode in mode_list:
    clip_list = glob.glob(os.path.join(root_path, mode, '*'))
    for clip_path in tqdm.tqdm(clip_list):
        if not os.path.isdir(clip_path): 
            continue
        if clip_path.split('/')[-1] in continue_list:
            continue
        
        clip_name = clip_path.split('/')[-1]
        out_dir_clip = os.path.join(out_path, mode, clip_name)
        if not os.path.exists(out_dir_clip):
            os.makedirs(out_dir_clip, exist_ok=True)
        
        frames_list = glob.glob(os.path.join(root_path, mode, clip_name, '*.png'))
        frames_list = sorted(frames_list)

        time_list = [(i+1)*(1./args.n) for i in range(args.n - 1)]
        with torch.no_grad():
            for ith_frame in tqdm.tqdm(range(len(frames_list)-1)):
            
                frame_name = frames_list[ith_frame].split('/')[-1].split('.')[0]

                gc.collect()
                torch.cuda.empty_cache()

                I0 = cv2.imread(frames_list[ith_frame])  
                I2 = cv2.imread(frames_list[ith_frame + 1])  
                
                I0_ = (torch.tensor(I0.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
                I2_ = (torch.tensor(I2.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)

                padder = InputPadder(I0_.shape, divisor=32)
                I0_, I2_ = padder.pad(I0_, I2_)

                # save original first frame
                cv2.imwrite(os.path.join(out_dir_clip, '%06d_%03d.png' % (int(frame_name), 0)), I0)
                
                images = []

                with torch.no_grad():
                    images = model.multi_inference(I0_, I2_, TTA=TTA, time_list=time_list, fast_TTA=TTA)

                # save interpolated frames
                for i, pred in enumerate(images):
                    images[i]  = ((padder.unpad(pred).detach().cpu().numpy().transpose(1, 2, 0) * 255.0)+0.5).astype(np.uint8)
                
                for idx in range(len(images)):
                    cv2.imwrite(os.path.join(out_dir_clip, '%06d_%03d.png' % (int(frame_name), time_list[idx] * 1000)), images[idx])

            I0 = cv2.imread(frames_list[-1])
            frame_name = frames_list[-1].split('/')[-1].split('.')[0]
            cv2.imwrite(os.path.join(out_dir_clip, '%06d_%03d.png' % (int(frame_name), 0)), I0)

# ==============================
# Split train clips into train / train1
# ==============================
train1_list = [
    "GOPR0374_11_00",
    "GOPR0374_11_02",
    "GOPR0374_11_03",
    "GOPR0378_13_00",
    "GOPR0379_11_00",
    "GOPR0380_11_00",
    "GOPR0384_11_01",
    "GOPR0372_07_00",
    "GOPR0372_07_01",
    "GOPR0374_11_01",
    "GOPR0857_11_00",
]

train_dir = os.path.join(out_path, 'train')
train1_dir = os.path.join(out_path, 'train1')
os.makedirs(train1_dir, exist_ok=True)

for clip_name in train1_list:
    src = os.path.join(train_dir, clip_name)
    dst = os.path.join(train1_dir, clip_name)

    if not os.path.exists(src):
        print(f'[Warning] Source clip does not exist: {src}')
        continue

    if os.path.exists(dst):
        print(f'[Warning] Destination already exists, skip: {dst}')
        continue

    shutil.move(src, dst)
    print(f'Moved: {clip_name} -> train1')
