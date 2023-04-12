import urllib.request
from data_process import *
import torch
from facenet_pytorch import InceptionResnetV1
from efficientnet_pytorch import EfficientNet
import gzip
from torchvision import models
from torch import nn
from Backbones.Margin.ArcMarginProduct import *
from Backbones.Margin.CosineMarginProduct import *
from Backbones.Margin.InnerProduct import *
from Backbones.Backbone import MobileFaceNet, CBAM
from Backbones.Backbone import iresnet


def initialize_model(model_name, num_classes=10575, feature_extract=True, use_pretrained=True):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = None
    input_size = 0
    if model_name == "FaceNet":
        """ InceptionResnetV1"""
        model = InceptionResnetV1(classify=True, pretrained='casia-webface', device=device).eval()
        # set_parameter_requires_grad(model, feature_extract)
        # in_channels = model.fc.in_features
        # model.fc = nn.Linear(in_features=in_channels, out_features=num_classes)
        # checkpoint = torch.load(r'E:\PaperLearning\PyProject\Cluster_test\resnet18_on_clustertest.pth')
        # model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        input_size = 160

    elif model_name == "cosface50":
        """ Alexnet"""
        # model = models.alexnet(pretrained=use_pretrained)
        # DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # # set_parameter_requires_grad(model, feature_extract)
        # in_channels = model.classifier[6].in_features
        # model.fc = nn.Linear(in_features=in_channels, out_features=num_classes)
        # model.eval().to(DEVICE)
        # input_size = 224
        model = iresnet.iresnet50(pretrained=False)
        DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # set_parameter_requires_grad(model, feature_extract)
        # in_channels = model.classifier[6].in_features
        # model.fc = nn.Linear(in_features=in_channels, out_features=num_classes)
        checkpoint = torch.load(r'C:\Codes\main_attack\attackface'
                                r'\Backbones\weight\glint360k_cosface_r50_fp16_0.1.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval().to(DEVICE)
        input_size = 112

    elif model_name == "arcface34":
        """ VGG19"""
        model = models.vgg19(pretrained=use_pretrained)
        DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # set_parameter_requires_grad(model, feature_extract)
        # in_channels = model.classifier[6].in_features
        # #model.classifier[6] = nn.Linear(in_channels,num_classes)
        # model.fc = nn.Linear(in_features=in_channels, out_features=num_classes)
        model.eval().to(DEVICE)
        input_size = 112

    elif model_name == "arcface50":
        # backbone
        model_name = 'ResNet50_IR'
        backbones = {'MobileFaceNet': MobileFaceNet.MobileFacenet(),
                     # 'ResNet50_IR':   CBAM.CBAMResNet(50, feature_dim=512, mode='ir'),
                     'ResNet50_IR':   iresnet.iresnet50(pretrained=False),
                     'SEResNet50_IR':  CBAM.CBAMResNet(50, feature_dim=512, mode='ir_se')}
        model_para_path = { # 'ResNet50_IR': r'C:\Codes\Pytorch_Face_Recognition--master\ResNet50-IR_net.pth',
                           'ResNet50_IR': r'..\main_attack\attackface\Backbones\weight\arcface_50.pth',
                           'SEResNet50_IR': r'..\main_attack\attackface\Backbones\weight'
                                            r'\CASIA_WebFace_SEResNet50_IR_net.pth',
                           'MobileFaceNet': r'..\main_attack\attackface\Backbones\weight'
                                            r'\CASIA_WebFace_MobileFaceNet_net.pth'}
        model = backbones[model_name]
        model_path = model_para_path[model_name]
        # load parameter
        model.load_state_dict(torch.load(model_path))
        model.eval().to(device)
        input_size = 112

    else:
        print("Invalid model name, exiting...")
        exit()
    return model, input_size

def loadImageNet224Model():
    model= models.resnet101(pretrained=True)
    model._fc = nn.Linear(in_features=2048, out_features=1000, bias=True)
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.eval().to(DEVICE)
    return model,224


def returnDimensions(dataset):
    if dataset == 'imagenet224':
        return 224, 224, 3
    elif dataset == 'CIFAR10':
        return 32, 32, 3
    elif 'casia-webface' in dataset or 'lfw' in dataset:
        return 112, 112, 3
    else:
        return None, None, None