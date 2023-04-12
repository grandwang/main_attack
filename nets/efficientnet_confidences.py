from torchvision import models, transforms
from PIL import Image
import torch
import os
import torch.nn.functional as nf
import efficientnet_pytorch
from torch import nn


def main(net, image, label, classes):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    input = torch.unsqueeze(transform(image), 0).to(DEVICE)
    out = net(input)
    _, index = torch.max(out, 1)

    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    _, indices = torch.sort(out, descending=True)
    prelist = [(classes[idx], percentage[idx].item()) for idx in indices[0][:-1]]
    print(prelist)
    if len(label) != 0:
        for idx in indices[0][:5]:
            if classes[idx] == label:
                return classes[idx], percentage[idx].item()
            else:

                atkpre = classes[index[0]]
                atkconfi = percentage[index[0]].item()
                return atkpre, atkconfi
    else:
        fstpre = classes[index[0]]
        fstconfi = percentage[index[0]].item()
        return fstpre, fstconfi,


if __name__ == '__main__':
    data_path = 'C:\\Codes\\datasets\\imagenet_mini\\train'
    input_path = data_path + '\\n01440764\\n01440764_15560.JPEG'
    image = Image.open(input_path).convert('RGB')
    classes = sorted(os.listdir(data_path))

    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # net = efficientnet_pytorch.EfficientNet.from_pretrained('efficientnet-b7')
    net = models.resnet50(pretrained=True)
    net.load_state_dict(torch.load('C:/Users/Wangxy/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth'))
    net._fc = nn.Linear(in_features=2048, out_features=len(classes), bias=True)

    # checkpoint = torch.load('../attacks/best-checkpoint-b7-1.bin')
    # net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    net.to(DEVICE)
    main(net, image, 'n01440764', classes)
