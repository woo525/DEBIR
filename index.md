---
layout: project_page
permalink: /
title: Dynamic Exposure Burst Image Restoration
authors:
    <A href="https://woo525.github.io/">Woohyeok Kim</A> <A href="https://rimchang.github.io/">Jaesung Rim</A> <A href="">Daeyeon Kim</A> <A href="https://www.scho.pe.kr/home">Sunghyun Cho</A>
affiliations:
    POSTECH
    <br><br><p style="font-style:italic;">The IEEE/CVF Computer Vision and Pattern Recognition (CVPR) 2026</p>
paper:
code: https://github.com/woo525/DEBIR
---

<!-- Using HTML to center the abstract -->
<div class="columns is-centered has-text-centered">
    <div class="column is-four-fifths">
        <h2>Abstract</h2>
        <div class="content has-text-justified">
Burst image restoration aims to reconstruct a high-quality image from burst images, which are typically captured using manually designed exposure settings.
Although these exposure settings significantly influence the final restoration performance, the problem of finding optimal exposure settings has been overlooked.
In this paper, we present Dynamic Exposure Burst Image Restoration (DEBIR), a novel burst image restoration pipeline that enhances restoration quality by dynamically predicting exposure times tailored to the shooting environment.
In our pipeline, Burst Auto-Exposure Network (BAENet) estimates the optimal exposure time for each burst image based on a preview image, as well as motion magnitude and gain.
Subsequently, a burst image restoration network reconstructs a high-quality image from burst images captured using these optimal exposure times.
For training, we introduce a differentiable burst simulator and a three-stage training strategy. 
Our experiments demonstrate that our pipeline achieves state-of-the-art restoration quality. 
Furthermore, we validate the effectiveness of our approach on a real-world camera system, demonstrating its practicality. 
        </div>
    </div>
</div>

---

## Method
![overview](/static/image/overview-1.png) <span style="color:gray"> *Overview of the proposed ParamISP framework. The full pipeline is constructed by combining learnable networks (ParamNet, LocalNet, GlobalNet) with invertible canonical camera operations (CanoNet). CanoNet consists of differentiable operations without learnable weights, where WB and CST denote white balance and color space transform, respectively.* </span>
<br/><br/>

![paramnet](/static/image/paramnet-1.png) <span style="color:gray"> *Architecture of ParamNet. (a) Given camera optical parameters, ParamNet estimates optical parameter features used for modulating the LocalNet and GlobalNet. (b) In order to deal with different scales and non-linearly distributed values of optical parameters, we propose to use non-linear equalization that exploits multiple non-linear mapping functions.* </span>
<br/><br/>

> Given a target camera, our goal is to learn its forward and inverse ISP processes that change with respect to camera parameters. To accomplish this, ParamISP is designed to have a pair of forward (RAW-to-sRGB) and inverse (sRGB-to-RAW) ISP networks. Both networks are equipped with ParamNet so that they adaptively operate based on camera parameters. In ParamISP, we classify camera parameters into two distinct categories: optical parameters (including exposure time, sensitivity, aperture size, and focal length) and canonical parameters (Bayer pattern, white balance coefficients, and a color correction matrix). To harness the canonical parameters, our ISP networks incorporate CanoNet, a subnetwork that performs canonical ISP operations without learnable weights. For the optical parameters, we introduce ParamNet, which is the key component to dynamically control the behavior of the ISP networks based on the optical parameters.

## Results
#### Qualitative
###### *1. Inverse (sRGB <span style="font-size:200%">&rarr;</span> RAW)*
![inverse](/static/image/inverse-1.png)

###### *2. Forward (RAW <span style="font-size:200%">&rarr;</span> sRGB)*
![forward](/static/image/forward-1.png)

#### Quantitative
![fwdinvQuan](/static/image/fwdinvQuan-1.png)

## Applications
#### *1. RAW Deblurring* 
![rawdeblur](/static/image/rawdeblur-1.png)

#### *2. Deblurring Dataset Synthesis*
![deblurdataset](/static/image/deblurdataset-1.png)

#### *3. HDR Reconstruction*
![hdr](/static/image/hdr-1.png)

#### *4. Camera-to-Camera Transfer*
![cam2cam](/static/image/cam2cam-1.png)

## Citation
```
@inproceedings{kim2026debir,
  title={Dynamic Exposure Burst Image Restoration},
  author={Kim, Woohyeok and Rim, Jaesung and Kim, Daeyeon and Cho, Sunghyun},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2026}
}
```
