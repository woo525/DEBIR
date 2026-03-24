---
layout: project_page
permalink: /
title: Dynamic Exposure Burst Image Restoration
authors:
    <A href="https://woo525.github.io/">Woohyeok Kim</A> &emsp; <A href="https://rimchang.github.io/">Jaesung Rim</A> &emsp; <A href="">Daeyeon Kim</A> &emsp; <A href="https://www.scho.pe.kr/home">Sunghyun Cho</A>
affiliations:
    <br>Pohang University of Science and Technology (POSTECH)
    <br><br><p style="font-style:italic;">The IEEE/CVF Computer Vision and Pattern Recognition (CVPR) 2026</p>
paper: https://woo525.github.io/DEBIR
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
![pipeline](/static/image/pipeline.png) <span style="color:gray"> *Overview of our pipeline. BAENet predicts the exposure times of each burst image from a preview image. Differentiable Burst Simulator generates burst images according to the exposure times. The restoration network then reconstructs a high-quality image from them. During inference, the simulator is removed, and the restoration network processes real burst images captured by our camera system.* </span>
<br/><br/>

> DEBIR enables effective burst image restoration by adaptively predicting an optimal exposure time for each burst image based on the shooting environment. 
To this end, DEBIR consists of a novel Burst Auto-Exposure Network (BAENet) and a burst image restoration network. 
BAENet determines the optimal exposure times for burst images, which maximize the restoration network performance, based on a preview image, current exposure settings, and motion information. 
The imaging system then captures burst images using these predicted exposure times, and the restoration network processes them to restore a clean, blur-free, and noise-free image.

## Analysis
![analysis](/static/image/analysis.png) <span style="color:gray"> *Analysis of predicted exposure times. (a) Scatter plots of exposure times of each frame and motion magnitude. 
(b) Exposure time histograms of the first frame for the minimum and maximum preview image gains. 
The unit of exposure times is 1/1920 sec., and preview image gain is normalized to [0,1].* </span>
<br/><br/>

## Results
#### Quantitative
![quantitative](/static/image/quantitative.png)

#### Qualitative
###### *1. Synthetic*
![qual_syn](/static/image/qual_syn.png)
<br/><br/>
![comparison_expbrk](/static/image/comparison_expbrk.png)

###### *2. Real-world*
![qual_real](/static/image/qual_real.png)


## Citation
```
@inproceedings{kim2026debir,
  title={Dynamic Exposure Burst Image Restoration},
  author={Kim, Woohyeok and Rim, Jaesung and Kim, Daeyeon and Cho, Sunghyun},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2026}
}
```
