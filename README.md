## Dynamic Exposure Burst Image Restoration<br><sub>Official PyTorch Implementation of the CVPR 2026 Paper</sub>

*Woohyeok Kim, Jaesung Rim, Daeyeon Kim, Sunghyun Cho<br>*

[\[Paper\]](https://arxiv.org/abs/2603.21784)
[\[Project Page\]](https://woo525.github.io/DEBIR/)

![overview](https://github.com/woo525/ParamISP/assets/32587029/b4bb291f-14e4-42dd-8642-518752843cc3)

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

      cd ../ # DEBIR
      conda env create -f install.yml
      conda init
      conda activate DEBIR
 
### Training
As described in the paper, ParamISP is trained in two stages for both the inverse and forward directions: pre-training and fine-tuning. Additionally, before applying it to applications, further joint fine-tuning can be conducted. We provide a small dataset example and the official weights reported in the paper to enable the execution of the code. You can set the dataset path through the **.env** file.

[\[Dataset example\]](https://drive.google.com/drive/folders/1ZCi3ZXLeM7Ary6eWlVaVTTDm-kWcXWjU?usp=sharing) [\[Official weights\]](https://drive.google.com/drive/folders/1ZCi3ZXLeM7Ary6eWlVaVTTDm-kWcXWjU?usp=sharing)
#### 1. Pre-training

        CUDA_VISIBLE_DEVICES=0 python models/paramisp.py -o demo --inverse train

#### 2. Fine-tuning
        
        CUDA_VISIBLE_DEVICES=0 python models/paramisp.py -o demo --inverse train --camera D7000

#### 3. Joint fine-tuning

        CUDA_VISIBLE_DEVICES=0,1 python models/paramisp_joint.py -o demo train --camera D7000 --pisp-inv weights/fine_tuning/inverse/D7000.ckpt --pisp-fwd weights/fine_tuning/forward/D7000.ckpt



### Test
 
        CUDA_VISIBLE_DEVICES=0 python models/paramisp.py -o demo --inverse test --ckpt weights/fine_tuning/inverse/D7000.ckpt --camera D7000

### Inference
 
        CUDA_VISIBLE_DEVICES=0 python models/paramisp.py -o demo --inverse predict --ckpt weights/fine_tuning/inverse/D7000.ckpt --camera D7000

### Citation
```
@inproceedings{kim2024paramisp,
  title={ParamISP: Learned Forward and Inverse ISPs using Camera Parameters},
  author={Kim, Woohyeok and Kim, Geonu and Lee, Junyong and Lee, Seungyong and Baek, Seung-Hwan and Cho, Sunghyun},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={26067--26076},
  year={2024}
}
```
