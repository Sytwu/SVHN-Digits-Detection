# NYCU Computer Vision 2025 Spring HW2
StudentID: 111550159 \
Name: Li-Zhong Szu-Tu (司徒立中)

## Introduction
The dataset is a subset from [Street View House Numbers (SVHN)](https://paperswithcode.com/dataset/svhn), consisting of 30,062 training images, 3,340 validation images, and 13,068 test images. Each image contains bounding boxes and corresponding labels for every digit present. The task is divided into two parts: the first involves detecting the class and bounding box of each digit in the image, while the second aims to predict the complete number represented by the detected digits. Performance is evaluated using mean Average Precision (mAP) for the first task and accuracy for the second. \
\
In this assignment, only the Faster R-CNN model is allowed; however, it is acceptable to modify its backbone, neck (Region Proposal Network), and head. In my implementation, I use ConvNeXt with pretrained weights as the backbone, and connect it to a Feature Pyramid Networks (a.k.a FPN) to fuse multi-scale features. \
\
The dataset can be downloaded [Here](https://drive.google.com/file/d/13JXJ_hIdcloC63sS-vF3wFQLsUP1sMz5/view)!

## How to install
How to install dependences
```
conda env create -f environment.yml
conda activate env
```

## How to run
How to execute the code
```
# Training
python main.py

# Inference
python infer.py
```
My model weights can be downloaded [Here](https://drive.google.com/drive/folders/1muKnj9P4nZVRsF5wNqhQohfj1RVsF9C4?usp=sharing)!

## Performance snapshot
A shapshot of the leaderboard
![image](https://github.com/user-attachments/assets/a655ff6e-f929-4a58-8d73-126824be431b)
\
Last Update: 2025/04/15 12:25 a.m. (GMT+8)
