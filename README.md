## Dynamic Exposure Burst Image Restoration<br><sub>Official PyTorch Implementation of the CVPR 2026 Paper</sub>

*Woohyeok Kim, Jaesung Rim, Daeyeon Kim, Sunghyun Cho<br>*

[\[Paper\]](https://arxiv.org/abs/2603.21784)
[\[Project Page\]](https://woo525.github.io/DEBIR/)

![overview](https://github.com/woo525/DEBIR/blob/gh-pages/static/image/pipeline.png)

### Abstract

Burst image restoration aims to reconstruct a high-quality image from burst images, which are typically captured using manually designed exposure settings. Although these exposure settings significantly influence the final restoration performance, the problem of finding optimal exposure settings has been overlooked. In this paper, we present Dynamic Exposure Burst Image Restoration (DEBIR), a novel burst image restoration pipeline that enhances restoration quality by dynamically predicting exposure times tailored to the shooting environment. In our pipeline, Burst Auto-Exposure Network (BAENet) estimates the optimal exposure time for each burst image based on a preview image, as well as motion magnitude and gain. Subsequently, a burst image restoration network reconstructs a high-quality image from burst images captured using these optimal exposure times. For training, we introduce a differentiable burst simulator and a three-stage training strategy. Our experiments demonstrate that our pipeline achieves state-of-the-art restoration quality. Furthermore, we validate the effectiveness of our approach on a real-world camera system, demonstrating its practicality.

### Dataset Preparation
* Ubuntu 20.04, Python 3.7.13, PyTorch 1.12.0

      cd ./EMA-VFI
      pip install -r requirements.txt

* Download GOPRO (GOPRO_Large_all) & RealBlur official dataset - [\[GOPRO\]](https://seungjunnah.github.io/Datasets/gopro.html) [\[RealBlur\]](https://github.com/rimchang/RealBlur?tab=readme-ov-file)

* Frame interpolation (x8)

      CUDA_VISIBLE_DEVICES=0 python interpolate_gopro_x8.py

* Synthesize dataset for stage-1 training

      CUDA_VISIBLE_DEVICES=0 python synthesize_dataset_for_stage1.py

### Environment Setting

      cd ../ # DEBIR/
      conda env create -f install.yml
      conda init
      conda activate DEBIR
 
### Training

[\[Dataset example\]](https://drive.google.com/drive/folders/1ZCi3ZXLeM7Ary6eWlVaVTTDm-kWcXWjU?usp=sharing) [\[Official weights\]](https://drive.google.com/drive/folders/1ZCi3ZXLeM7Ary6eWlVaVTTDm-kWcXWjU?usp=sharing)


        # [Stage-1]
        CUDA_VISIBLE_DEVICES=0,1,2,3 python train_stage1.py
        
        # [Stage-2]
        CUDA_VISIBLE_DEVICES=0 python prepare_stage2w.py # make pseudo-gt 
        CUDA_VISIBLE_DEVICES=0,1,2,3 python train_stage2w.py # stage-2 warm-up
        CUDA_VISIBLE_DEVICES=0,1,2,3 python train_stage2m.py # stage-2 main
        
        # [Stage-3]
        CUDA_VISIBLE_DEVICES=0,1,2,3 python train_stage3.py

### Test
 
        CUDA_VISIBLE_DEVICES=0 python test.py

### Citation
```
@inproceedings{kim2026debir,
  title={Dynamic Exposure Burst Image Restoration},
  author={Kim, Woohyeok and Rim, Jaesung and Kim, Daeyeon and Cho, Sunghyun},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2026}
}
```
