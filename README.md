# MISPSO-Attack

This is the Pytorch code for our paper “MISPSO-Attack: An Efficient Adversarial Watermarking Attack Based on Multiple Initial Solution Particle Swarm Optimization”.

![Algprocessalter](https://user-images.githubusercontent.com/36922651/231676550-a22605ae-04df-483b-a2ec-54d2d451c158.png)

Pretrained weights at Aliyun Drive: https://www.aliyundrive.com/s/UcKWpgwtJeu

All datasets' url have been marked in our manuscript.

IF you download imagenet or imagenet-mini (https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000), pls change the "data_dir" to your path.
Every datasets should have a same structure like this:

```
dataset
-dataset name
 --class 1
   ---pic001
   ---pic002
   ---pic003  
```

## Quick Start

Running this command for attacks:

```
python advattack_on_mini.py
```

Running this command for attack F-R task:

```
python advattack_on_face.py
```

### Environment Settings:
This project is tested under the following environment settings:
- OS: win10 21h2
- advertorch=0.2.3
  cudatoolkit=11.3.1
  dlib=19.23.1
  numpy=1.16.6
  opencv=3.4.2
  opencv-contrib-python=3.4.2.16
  opencv-python=3.4.2.16
  pandas=1.2.0
  python=3.7.11
  pytorch=1.11.0=py3.7_cuda11.3_cudnn8_0
  scikit-learn=1.0.1
