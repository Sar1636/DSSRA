# Dual Spectral-Spatial Residual Adaptive Network for Hyperspectral Image Classification in the Presence of Noisy Labels

# Introduction
In real-world scenarios, Hyperspectral Image (HSI) datasets introduce potential noise inaccuracies due to multiple annotators. Label noise poses a significant challenge for practical deep learning, yet this issue is largely unexplored. Existing methods, which attempt to clean the noisy labelled data to increase classification accuracy, are computationally expensive and face the risk of removing correctly labelled data. In contrast, other methods that work with noisy labelled data but attempt to minimize the noise impact on classification by formulating a robust loss function lose classification accuracy when the ratio of incorrectly to correctly labelled data is high. This work proposes a Dual Spectral-Spatial Residual Adaptive (DSSRA) network to minimize the noise effect even when the amount of noisy labelled data is high. It offers the following contributions: (1) effective salient feature extraction modules to enhance the discriminatory representation of different classes in the proposed DSSRA network; (2) an adjusted noise tolerance loss (ANTL) function that down-weights the impact of learning with noisy labels. ANTL combines normalized focal loss and reverse cross-entropy to counter label noise; and (3) extensive testing on noisy versions of several benchmark HSI datasets. Results show that our DSSRA model outperforms state-of-the-art HSI classification methods in handling noisy labels, offering a robust solution for real-world applications.
A python deep learning model which is super useful in dealing with hyperspectral image classification with noisy labels.

- flowchart
  [DSSRA.pdf](https://github.com/user-attachments/files/15927960/DSSRA.pdf)

# Description

The loss function gets its code from [Link](https://github.com/HanxunH/Active-Passive-Losses)

Our code is based on [link](https://github.com/Li-ZK/DCRN-2021)

# Requirement

- Python >= 3.11.0
- torch >= 2.3.0
- PyTorch 2.3.0
- torchvision 0.18.0
- CUDA 11.8
- tqdm

# Usage

### Run with default settings

Run ` python main.py` for training and test.

Download the dataset from the [link](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes)

Example to run with personal settings:

`python main.py --batch_size 32 --max_iter 200`

An example dataset folder has the following structure:

```
datasets
├── Pavia
│   ├── Pavia.mat
│   ├── Pavia_gt.mat
├── Dioni
│   ├── Dioni.mat
│   └── Dioni_gt_out68.mat
├── Botswana
│   ├── Botswana.mat
│   ├── Botswana_gt.mat
└── paviaU
    ├── paviaU_gt.mat
    └── paviaU.mat
```
