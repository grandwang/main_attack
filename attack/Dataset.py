from data_process import *
import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
from efficientnet_pytorch import EfficientNet
from torch import nn

def returnDimensions(dataset):
    if dataset == 'imagenet224':
        return 224, 224, 3
    elif dataset == 'CIFAR10':
        return 32, 32, 3
    else:
        return None, None, None


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    model = None
    input_size = 0
    if model_name == "resnet":
        """ resnet18,resnet34,resnet50,resnet101"""
        # model = models.resnet18(pretrained=use_pretrained)
        model = models.resnet50(pretrained=True)
        # set_parameter_requires_grad(model, feature_extract)
        # in_channels = model.fc.in_features
        # model.fc = nn.Linear(in_features=in_channels, out_features=num_classes)
        # checkpoint = torch.load(r'E:\PaperLearning\PyProject\Cluster_test\resnet18_on_clustertest.pth')
        # model.load_state_dict(checkpoint['model_state_dict'])
        DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.eval().to(DEVICE)
        input_size = 224

    elif model_name == "GoogLeNet":
        """ GoogLeNet"""
        model = models.googlenet(pretrained=True)
        DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # set_parameter_requires_grad(model, feature_extract)
        # in_channels = model.classifier[6].in_features
        model.eval().to(DEVICE)
        input_size = 224

    elif model_name == "efficientnet":
        """ efficientnet-b0"""
        model = EfficientNet.from_pretrained('efficientnet-b0')
        DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # set_parameter_requires_grad(model, feature_extract)
        # in_channels = model.classifier[6].in_features
        model.eval().to(DEVICE)
        input_size = 224

    elif model_name == "vgg":
        """ VGG19"""
        model = models.vgg19(pretrained=use_pretrained)
        DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # set_parameter_requires_grad(model, feature_extract)
        in_channels = model.classifier[6].in_features
        # model.classifier[6] = nn.Linear(in_channels,num_classes)
        model.fc = nn.Linear(in_features=in_channels, out_features=num_classes)
        model.eval().to(DEVICE)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet1.0"""
        model = models.squeezenet1_0(pretrained=use_pretrained)
        DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # set_parameter_requires_grad(model, feature_extract)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model.num_classes = num_classes
        model.eval().to(DEVICE)
        input_size = 224

    elif model_name == "densenet":
        """ Densenet"""
        model = models.densenet121(pretrained=use_pretrained)
        DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # set_parameter_requires_grad(model, feature_extract)
        in_channels = model.classifier.in_features
        model.fc = nn.Linear(in_features=in_channels, out_features=num_classes)
        model.eval().to(DEVICE)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
             Be careful, expects (299,299) sized images and has auxiliary output
        """
        model = models.inception_v3(pretrained=use_pretrained)
        DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # set_parameter_requires_grad(model, feature_extract)
        # 处理辅助网络
        in_channels = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(in_channels, num_classes)
        # 处理主要网络
        in_channels = model.fc.in_features
        model.fc = nn.Linear(in_features=in_channels, out_features=num_classes)
        model.eval().to(DEVICE)
        input_size = 299
    else:
        print("Invalid model name, exiting...")
        exit()
    return model, input_size
