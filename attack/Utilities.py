import numpy
import torch
import numpy as np
import torch.nn.functional
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
from nets.resnet101_confidences import main as ec_main
from data_process import cal_sigema, sigfun, sigfun_quo
from copy import deepcopy


def softmax(x):
    x = [709 if i > 709 else i for i in x]
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def getpred(path, label, model, classes):
    img = Image.open(path).convert('RGB')
    predict, _ = ec_main(model, img, label, classes)

    return predict


def object_function(model, input):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    pred = model(torch.unsqueeze(transform(input.convert('RGB')), dim=0).cuda()).cpu().detach().numpy().astype(
        np.float64)
    if not np.round(np.sum(pred[0]), decimals=2) == 1.0 or (
            np.round(np.sum(pred[0]), decimals=2) == 1.0 and any(n < 0 for n in pred[0])):
        pred[0] = softmax(pred[0])
    return pred


def predictSample(net, input):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    input = Variable(torch.unsqueeze(transform(input), dim=0), requires_grad=False).cuda()
    out = net(input)
    pred = out.cpu().detach().numpy()
    if not np.round(np.sum(pred[0]), decimals=2) == 1.0 or (
            np.round(np.sum(pred[0]), decimals=2) == 1.0 and any(n < 0 for n in pred[0])):
        pred[0] = softmax(pred[0])
    return pred


def Open_logo(logopath, sl, dataset):
    watermark = Image.open(logopath).convert('RGBA')
    if dataset == 'imagenet224':
        size = 224
    if dataset == 'CIFAR10':
        size = 32
    elif dataset == 'MNIST':
        size = 28
    watermark_x1, watermark_y1 = watermark.size
    print("watermark.size", watermark.size)
    watermark_scale = min(size / (sl * watermark_x1),
                          size / (sl * watermark_y1))
    watermark_x1 = int(watermark_x1 * watermark_scale)
    watermark_y1 = int(watermark_y1 * watermark_scale)
    return watermark, watermark_x1, watermark_y1, size


def add_watermark_to_image(image, xs, watermark, sl):
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224)
    ])
    image = preprocess(image)
    rgba_image = image.convert('RGBA')
    rgba_watermark = watermark.convert('RGBA')

    image_x, image_y = rgba_image.size
    watermark_x, watermark_y = rgba_watermark.size

    scale = sl
    watermark_scale = min(image_x / (scale * watermark_x), image_y / (scale * watermark_y))

    new_size = (int(watermark_x * watermark_scale), int(watermark_y * watermark_scale))

    rgba_watermark = rgba_watermark.resize(new_size, resample=Image.ANTIALIAS)
    temp = rgba_watermark.convert('L')
    rgba_watermark_mask = temp.point(lambda p: min(p, int(xs[0])))

    rgba_watermark.putalpha(rgba_watermark_mask)

    watermark_x, watermark_y = rgba_watermark.size
    a = np.array(xs[1])

    a = np.clip(a, 0, 224 - watermark_x)
    b = np.array(xs[2])
    b = np.clip(b, 0, 224 - watermark_y)

    x_pos = int(a)
    y_pos = int(b)
    rgba_image.paste(rgba_watermark, (x_pos, y_pos), rgba_watermark_mask)
    return rgba_image


def compareImages(image1, image2, dLength):
    diff = np.sum(np.square(np.subtract(image1, image2))) / dLength
    return diff


def halfsOfN(n, limit):
    lst = [n]
    while n / 2 >= limit:
        n = int(n / 2)
        lst.append(n)
    return lst


def splitSoluIntoNbyNRegions(numofMethods, numberOfParticles, channels, k):
    indArray = []
    ind = 0
    for i in range(numofMethods):
        colArray = []
        for j in range(numberOfParticles):
            channelArray = []
            for c in range(channels):
                channelArray.append(ind)
                ind = ind + 1
            colArray.append(channelArray)
        indArray.append(colArray)
    chunks = []
    for rows in range(0, numofMethods, k):
        if rows + k > numofMethods:
            r = indArray[rows:]
        else:
            r = indArray[rows:rows + k]
        if r:
            for cols in range(0, numberOfParticles, k):
                chunks.append([])
                for rr in r:
                    if cols + k > numberOfParticles:
                        ch = [s for c in rr[cols:] for s in c]
                    else:
                        ch = [s for c in rr[cols:cols + k] for s in c]
                    if ch:
                        chunks[-1].extend(ch)
    return chunks


def re2Watermark(sample, dataset, img, watermark, sl, numofMethods, numberOfParticles, input_size):
    data_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(), ])
    imglist = []
    img = img.convert('RGB')
    newSample = None
    if 'CIFAR10' in dataset:
        newSample = np.reshape(sample, (32, 32, 3))
    elif 'imagenet224' in dataset:
        sample = np.reshape(sample, (numofMethods * numberOfParticles, 3))
        for i in range(0, len(sample)):
            Que = sample[i]
            attack_image = add_watermark_to_image(img, Que, watermark, sl).convert('RGB')
            attack_image_ele = data_transform(attack_image)
            imglist.append(attack_image_ele)
    return imglist


def re2Image(sample, dataset, img, watermark, sl):
    if 'CIFAR10' in dataset:
        newSample = np.reshape(sample, (32, 32, 3))
    elif 'imagenet224' in dataset:
        attack_image = add_watermark_to_image(img, sample, watermark, sl).convert('RGB')
    return attack_image


def rangebound(c, lower, upper):
    for i in range(len(c)):
        c[i] = np.clip(c[i], lower[i % 3], upper[i % 3])
    return c


def fitnessScore(self, net, xs, input, classes, watermark, sl, j):
    fitscore = []
    newPred = []
    alpha = 0.9
    beta = 1 - alpha
    img = input.convert('RGB')
    _, fstconfi = ec_main(net, img, '', classes)

    data_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(), ])
    imglist = []

    for i in range(len(xs)):
        queries = xs[i]
        attack_image = add_watermark_to_image(img, queries, watermark, sl).convert('RGB')
        attack_image_ele = data_transform(attack_image)
        imglist.append(attack_image_ele)
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batchimage_ele = torch.stack(imglist, dim=0)
    net.eval()

    with torch.no_grad():
        output = net(batchimage_ele.to(DEVICE)).cpu()
    prelist, index = torch.max(output, dim=1)
    percentage = torch.nn.functional.softmax(output, dim=1)
    _, indices = torch.sort(output, descending=True)
    indicesnp = indices.numpy()

    for idx in index.numpy():
        if idx != j:
            temp = 0
            initSS = ''
            for v in range(len(prelist)):
                if (index.numpy()[v] != j) and prelist.numpy()[v] > temp:
                    temp = prelist.numpy()[v]
                    idx = index.numpy()[v]
                    initSS = xs[v]
            return True, (idx, initSS)
    i = 0
    for idx in indicesnp[:, 0]:
        fitscore.append(percentage[i][idx].item())
        newPred.append(idx)
        i = i + 1
    return fitscore, newPred


def InitFitnessScore(self, net, xs, input, classes, watermark, sl, j):
    fitscore = []
    newPred = []
    alpha = 0.9
    beta = 1 - alpha
    img = input.convert('RGB')
    _, fstconfi = ec_main(net, img, '', classes)

    data_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(), ])
    imglist = []
    for i in range(len(xs)):
        queries = xs[i]
        attack_image = add_watermark_to_image(img, queries, watermark, sl).convert('RGB')
        attack_image_ele = data_transform(attack_image)
        imglist.append(attack_image_ele)
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batchimage_ele = torch.stack(imglist, dim=0)
    net.eval()
    with torch.no_grad():
        output = net(batchimage_ele.to(DEVICE)).cpu()
    _, index = torch.max(output, dim=1)

    percentage = torch.nn.functional.softmax(output, dim=1)
    _, indices = torch.sort(output, descending=True)
    indicesnp = indices.numpy()
    i = 0
    for idx in indicesnp[:, 0]:
        fitscore.append(percentage[i][idx].item())
        newPred.append(idx)
        i = i + 1
    return numpy.array(fitscore), numpy.array(newPred)


def drawIterline(percentage, indicesnp, j):
    oriclass_ind = percentage.data.numpy()[:, j]
    oriclass_rate = oriclass_ind[np.argmax(oriclass_ind)]

    atkclass_temp = []
    atkclass_ind = deepcopy(indicesnp)
    atkclass_ind = atkclass_ind[:, 0]
    for d in range(len(atkclass_ind)):
        if atkclass_ind[d] != j:
            atkclass_temp.append(percentage.data.numpy()[d, atkclass_ind[d]])
        else:
            atkclass_temp.append(0)
    atkclass_rate = atkclass_temp[np.argmax(atkclass_temp)]
    return oriclass_rate, atkclass_rate
