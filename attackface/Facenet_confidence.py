import os
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.nn import functional
from PIL import Image

def main(net, img, label, classes):
    # If required, create a face detection pipeline using MTCNN:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(image_size=160, device=device)
    # Get cropped and prewhitened image tensor
    img_cropped = mtcnn(img)
    if img_cropped is None:
        return 'error', ''
    else:
        out = net(img_cropped.unsqueeze(0).to(device))
        _, index = torch.max(out, 1)
        percentage = torch.nn.functional.softmax(out, dim=1)[0] #对最大预测值的位置进行softmax
        fstpre = classes[index[0]]
        fstconfi = percentage[index[0]].item()
        return fstpre, fstconfi