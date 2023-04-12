from torchvision import models,transforms
from PIL import Image
import torch
import os
import numpy as np
import torch.nn.functional as nf
import efficientnet_pytorch
from torch.autograd import Variable
from torch import nn

def main(net,image,label,classes):
    transform = transforms.Compose([
        # transforms.RandomRotation(10),  # rotate +/- 10 degrees
        # transforms.RandomHorizontalFlip(),  # reverse 50% of images
        transforms.Resize(224),  # resize shortest side'
        transforms.CenterCrop(224),  # crop longest side
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406],
        #                       [0.229, 0.224, 0.225])
    ])
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    #input = torch.unsqueeze(transform(image),0).to(DEVICE)
    input = Variable(torch.unsqueeze(transform(image), dim=0), requires_grad=False).to(DEVICE)
    # print(net(input.cuda()).cpu().detach().numpy())
    out = net(input)
    _, index = torch.max(out, 1)
    percentage = torch.nn.functional.softmax(out, dim=1)[0]
    _, indices = torch.sort(out, descending=True)
    # prelist = [(classes[idx], percentage[idx].item()) for idx in indices[0][:5]]
    # print(prelist)
    if len(label)!=0:
        for idx in indices[0][:3]:
            if classes[idx] == label:
                return classes[idx], percentage[idx].item()
            else:
                #print(classes[index[0]], percentage[index[0]].item(),label)
                atkpre=classes[index[0]]
                atkconfi = percentage[index[0]].item()
                return atkpre, atkconfi
    else:
        fstpre = classes[index[0]]
        fstconfi = percentage[index[0]].item()
        return fstpre, fstconfi