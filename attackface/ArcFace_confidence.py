import os
from PIL import Image
from torch.nn import functional
from torchvision.transforms import transforms
from Backbones.Margin.ArcMarginProduct import *
from Backbones.Margin.CosineMarginProduct import *
from Backbones.Margin.InnerProduct import *
from Backbones.Backbone import MobileFaceNet,CBAM

def arc_main(net, image, judgeflag, classes):
    transform = transforms.Compose([
        transforms.Resize(112),
        transforms.ToTensor()
    ])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print('extracing deep features from the face pair {}...'.format(count))

    # margin
    margin_path = r'..\attackface\Backbones\weight\ResNet50-IR_margin.pth'
    margin = ArcMarginProduct(512, len(classes), s=32.0)
               # 'CosFace': CosineMarginProduct(512, len(classes), s=32.0),
               # 'Softmax': InnerProduct(512, len(classes))}
    margin.load_state_dict(torch.load(margin_path))
    margin.eval()
    margin.to(device)
    data = transform(image).to(device)
    # label 可以随意指定，不影响结果
    label = torch.tensor(int(100)).to(device)
    with torch.no_grad():
        embeddings = net(data.unsqueeze(0))
        if embeddings is None:
            return 'error', ''
        out = margin(embeddings, label)
    _, index = torch.max(out, 1)
    if judgeflag == 'full':
        return out, _
    percentage = torch.nn.functional.softmax(out, dim=1)[0]  # 对最大预测值的位置进行softmax
    fstpre = classes[index[0]]
    fstconfi = percentage[index[0]].item()
    return fstpre, fstconfi

def cos_main(net, image, judgeflag, classes):
    transform = transforms.Compose([
        transforms.Resize(112),
        transforms.ToTensor()
    ])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # margin
    margin_path = r'..\attackface\Backbones\weight\CBAM_CosFace_margin.pth'

    margin = CosineMarginProduct(512, len(classes), s=32.0)
    margin.load_state_dict(torch.load(margin_path))
    margin.eval()
    margin.to(device)
    data = transform(image).to(device)
    #label 可以随意指定，不影响结果
    label = torch.tensor(int(100)).to(device)
    with torch.no_grad():
        embeddings = net(data.unsqueeze(0))
        if embeddings is None:
            return 'error', ''
        out = margin(embeddings, label)
    _, index = torch.max(out, 1)
    if judgeflag == 'full':
        return out, _
    percentage = torch.nn.functional.softmax(out, dim=1)[0]  # 对最大预测值的位置进行softmax
    fstpre = classes[index[0]]
    fstconfi = percentage[index[0]].item()
    return fstpre, fstconfi