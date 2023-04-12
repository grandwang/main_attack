import os
import numpy
import torch
import numpy as np
import torch.nn.functional
from PIL import Image
from torchvision import transforms, datasets
from torch.autograd import Variable
from ArcFace_confidence import arc_main, cos_main
from data_process import cal_sigema, sigfun, sigfun_quo
from facenet_pytorch import MTCNN
from copy import deepcopy
from torch.nn import functional
from Backbones.Margin.ArcMarginProduct import *
from Backbones.Margin.CosineMarginProduct import *
from Backbones.Margin.InnerProduct import *
from Backbones.Backbone import MobileFaceNet, CBAM
from tqdm import tqdm, trange
from mtcnn_pytorch_master.test import crop_face
import joblib
from torch.utils.data import DataLoader
import shutil

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=112, device=device)

margin_path = r'..\main_attack\attackface\Backbones\weight\CBAM_CosFace_margin.pth'
margin_type = 'CosFace'
margins = {'ArcFace': ArcMarginProduct(512, 10575, s=32.0),
           'CosFace': CosineMarginProduct(512, 10575, s=32.0),
           'Softmax': InnerProduct(512, 10575)}
margin = margins[margin_type]
margin.load_state_dict(torch.load(margin_path))
margin.eval()
margin.to(device)


def crop_imgs(imgs, w, h):
    trans = transforms.Compose([
        transforms.Resize(112),
        transforms.ToTensor(),

    ])
    crops_result = []
    crops_tensor = []
    if len(imgs) > 1:
        for i in range(len(imgs)):
            crop = crop_face(imgs[i], w, h)
            crop_ts = trans(crop)
            crops_result.append(crop)
            crops_tensor.append(crop_ts)
    else:
        for i in range(len(imgs)):
            crop = crop_face(imgs[i], w, h)
            crop_ts = trans(crop)
            crops_result.append(crop)
            crops_tensor.append(crop_ts)
    return crops_result, crops_tensor


def cosin_all(feature, model_name, device):
    embedding_sets = joblib.load(r'../attackface/stmodels/'
                                 r'{}/embeddings_{}.pkl'.format(model_name, model_name))
    sets = torch.t(embedding_sets).to(device)
    numerator = torch.mm(feature, sets)
    norm_x1 = torch.norm(feature, dim=1)
    norm_x1 = torch.unsqueeze(norm_x1, 1)
    norm_x2 = torch.norm(sets, dim=0)
    norm_x2 = torch.unsqueeze(norm_x2, 0)

    denominator = torch.mm(norm_x1, norm_x2)
    metrics = torch.mul(numerator, 1 / denominator)
    return metrics.cpu().detach()


def check_all(adv_face_ts, threat, threat_name, device):
    percent = []
    typess = []

    def collate_fn(x):
        return x

    loader = DataLoader(
        adv_face_ts,
        batch_size=55,
        shuffle=False,
        collate_fn=collate_fn
    )

    for X in loader:

        advface_ts = torch.stack(X).to(device)
        X_op = nn.functional.interpolate(advface_ts, (112, 112),
                                         mode='bilinear', align_corners=False)
        feature = threat(X_op)
        for i in range(len(feature)):
            sim_all = cosin_all(torch.unsqueeze(feature[i], 0), threat_name, device)
            _, indices = torch.sort(sim_all, dim=1, descending=True)
            cla = [indices[0][0].item(), indices[0][1].item(), indices[0][2].item(),
                   indices[0][3].item(), indices[0][4].item(), indices[0][5].item(), indices[0][6].item()]
            typess.append(cla)
            tage = sim_all[0].numpy()
            percent.append(tage)
    return typess, np.array(percent)


def softmax(x):
    # TO PREVENT OVERFLOWING OF EXP
    x = [709 if i > 709 else i for i in x]
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def getpred(path, label, model, classes):  # (path,label,sl,np_list[ii]))
    img = Image.open(path).convert('RGB')
    if mtcnn(img) is None:
        return 'error on MTCNN'
    img_cropped = mtcnn(img).to(device)
    out = model(img_cropped.unsqueeze(0))
    _, index = torch.max(out, 1)

    fstpre, _ = cos_main(model, img, '', classes)
    return fstpre


def object_function(model, input):
    transform = transforms.Compose([
        transforms.Resize(112),
        transforms.ToTensor()
    ])
    data = transform(input).to(device)

    label = torch.tensor(int(100)).to(device)
    with torch.no_grad():
        embeddings = model(data.unsqueeze(0))
        out = margin(embeddings, label)
    pred = out.cpu().detach().numpy().astype(np.float64)
    if not np.round(np.sum(pred[0]), decimals=2) == 1.0 or (
            np.round(np.sum(pred[0]), decimals=2) == 1.0 and any(n < 0 for n in pred[0])):
        pred[0] = softmax(pred[0])
    return pred


def Open_logo(logopath, sl, dataset):
    watermark = Image.open(logopath)
    if dataset == 'casia-webface' or dataset == 'lfw':
        size = 112
    if dataset == 'vggface2':
        size = 112
    watermark_x1, watermark_y1 = watermark.size
    print("watermark.size", watermark.size)
    watermark_scale = min(size / (sl * watermark_x1),
                          size / (sl * watermark_y1))
    watermark_x1 = int(watermark_x1 * watermark_scale)
    watermark_y1 = int(watermark_y1 * watermark_scale)
    return watermark, watermark_x1, watermark_y1, size


def add_watermark_to_image(image, xs, watermark, sl):
    preprocess = transforms.Compose([
        transforms.Resize(112),
        transforms.CenterCrop(112)
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

    rgba_watermark_mask = rgba_watermark.convert("L").point(lambda x: min(x, int(xs[0])))

    rgba_watermark.putalpha(rgba_watermark_mask)

    watermark_x, watermark_y = rgba_watermark.size
    a = np.array(xs[1])
    a = np.clip(a, 0, 112 - watermark_x)
    b = np.array(xs[2])
    b = np.clip(b, 0, 112 - watermark_y)

    x_pos = int(a)
    y_pos = int(b)
    rgba_image.paste(rgba_watermark, (x_pos, y_pos), rgba_watermark_mask)
    return rgba_image


def compareImages(image1, image2, dLength):
    diff = np.sum(np.square(np.subtract(image1, image2))) / dLength  # L2
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
    imglist = []
    img = img.convert('RGB')
    newSample = None
    if 'CIFAR10' in dataset:
        newSample = np.reshape(sample, (32, 32, 3))
    elif 'casia-webface' in dataset or 'vggface2' in dataset or 'lfw' in dataset:
        sample = np.reshape(sample, (numofMethods * numberOfParticles, 3))
        for i in range(0, len(sample)):
            Que = sample[i]
            attack_image = add_watermark_to_image(img, Que, watermark, sl).convert('RGB')
            attack_image_ele = mtcnn(attack_image)
            imglist.append(attack_image_ele)
    return imglist


def re2Image(sample, dataset, img, watermark, sl):
    img = img.convert('RGB')
    if 'CIFAR10' in dataset:
        attack_image = np.reshape(sample, (32, 32, 3))
    elif 'casia-webface' in dataset or 'vggface2' in dataset or 'lfw' in dataset:
        attack_image = add_watermark_to_image(img, sample, watermark, sl).convert('RGB')
    return attack_image


def rangebound(c, lower, upper):
    for i in range(len(c)):
        c[i] = np.clip(c[i], lower[i % 3], upper[i % 3])
    return c


def fitnessScore(self, model, xs, input, label, watermark, sl, j):
    fitscore = []
    newPred = []
    img = input.convert('RGB')
    imglist = []
    transform = transforms.Compose([
        transforms.Resize(112),
        transforms.ToTensor()
    ])
    shutil.rmtree('./fts')
    os.makedirs("fts/{}".format(label))
    if not os.path.exists("fts/{}".format(label)):
        os.makedirs("fts/{}".format(label))
    for i in range(len(xs)):
        queries = xs[i]
        attack_image = add_watermark_to_image(img, queries, watermark, sl).convert('RGB')
        attack_image.save('fts/{}/{}.png'.format(label, i))
        # attack_image_ele = mtcnn(attack_image)
        attack_image_ele = transform(attack_image)
        if attack_image_ele is None:
            return False, 'MTCNN has failed to get key point'
        imglist.append(attack_image_ele)

    fts_data = datasets.ImageFolder('fts')
    batchimage_ele = []
    for i in range(len(fts_data)):
        batchimage_ele.append(fts_data[i][0])
    # batchimage_ele = torch.stack(imglist, dim=0)
    # net.eval()
    # data = batchimage_ele.to(device)
    # labels = torch.tensor(np.zeros(len(xs), dtype="int64")).to(device)
    with torch.no_grad():
        out_result, out_tensor = crop_imgs(batchimage_ele, 112, 112)
        # embeddings = model(batchimage_ele.to(DEVICE))
        sim_labels, sim_probs = check_all(out_tensor, model, 'cosface50', device)
    # with torch.no_grad():
    #     output = net(batchimage_ele.to(device)).cpu()
    # _, index = torch.min(torch.tensor(sim_probs), 1)
    # _, indices = torch.sort(torch.tensor(sim_probs), descending=False)
    # # low_con = output.numpy()
    # low_con = sim_probs
    # # low_con = low_con[:, start_label[0]]
    # low_con = low_con[:, start_label[0]]
    # low_con_idx = np.argmin(low_con, axis=0)
    # # 找出最小的atkconfi对应的xs值
    # minxs = xs[low_con_idx]
    # cost = low_con[low_con_idx]

    prelist, index = torch.max(torch.tensor(sim_probs), 1)  # ("索引最大预测值的位置",index)
    percentage = torch.nn.functional.softmax(torch.tensor(sim_probs), dim=1)
    tep, indices = torch.sort(torch.tensor(sim_probs), descending=True)

    indicesnp = indices.numpy()

    true_label_idx = []
    for line in indicesnp:
        true_label_idx.append(np.where(line == j)[0])
    # 如果当前在原始分类下，出现错误索引，成功
    # print(true_label_idx[j][0])
    for idx in index.numpy():
        if idx != j:

            temp = 0
            initSS = ''
            for v in range(len(prelist)):

                if (index.numpy()[v] != j) and ((tep.numpy()[v][idx] - tep.numpy()[v][true_label_idx[v][0]]) > 0.15):
                    print("Confidence distance", tep.numpy()[v][idx] - tep.numpy()[v][true_label_idx[v][0]])
                    temp = prelist.numpy()[v]
                    idx = index.numpy()[v]
                    initSS = xs[v]  # 初始成功效果最好的解
                    return True, (idx, initSS)
                else:
                    continue
    i = 0
    for idx in indicesnp[:, 0]:
        fitscore.append(percentage[i][idx].item())
        newPred.append(idx)
        i = i + 1
    shutil.rmtree('./fts')
    os.mkdir('./fts')
    return fitscore, newPred


def InitFitnessScore(self, net, xs, input, classes, watermark, sl, j):
    fitscore = []
    newPred = []
    alpha = 0.9
    beta = 1 - alpha
    img = input.convert('RGB')
    data_transform = transforms.Compose(
        [transforms.Resize(112),
         transforms.ToTensor()])
    shutil.rmtree('./initfts')
    os.makedirs("initfts/{}".format(j))
    if not os.path.exists("initfts/{}".format(j)):
        os.makedirs("initfts/{}".format(j))
    imglist = []
    for i in range(len(xs)):
        queries = xs[i]
        attack_image = add_watermark_to_image(img, queries, watermark, sl).convert('RGB')
        attack_image.save('initfts/{}/{}.png'.format(j, i))
        attack_image_ele = data_transform(attack_image)
        if attack_image_ele is None:
            return False, 'MTCNN has failed to get key point'
        imglist.append(attack_image_ele)

    fts_data = datasets.ImageFolder('initfts')
    batchimage_ele = []
    for i in range(len(fts_data)):
        batchimage_ele.append(fts_data[i][0])
    with torch.no_grad():
        out_result, out_tensor = crop_imgs(batchimage_ele, 112, 112)

        sim_labels, sim_probs = check_all(out_tensor, net, 'cosface50', device)

    # prelist, index = torch.max(output, dim=1)  # ("索引最大预测值的位置",index)
    prelist, index = torch.max(torch.tensor(sim_probs), 1)
    percentage = torch.nn.functional.softmax(torch.tensor(sim_probs), dim=1)
    tep, indices = torch.sort(torch.tensor(sim_probs), descending=True)
    indicesnp = indices.numpy()
    # true_label_idx=[]
    # for line in indicesnp:
    #     true_label_idx.append(np.where(line == j)[0])
    i = 0
    for idx in indicesnp[:, 0]:
        fitscore.append(percentage[i][idx].item())
        newPred.append(idx)
        i = i + 1
    shutil.rmtree('./initfts')
    os.mkdir('./initfts')
    return numpy.array(fitscore), numpy.array(newPred)


def predictSample(net, input):
    transform = transforms.Compose([
        transforms.Resize(112),
        transforms.ToTensor()
    ])
    data = transform(input).to(device)
    label = torch.tensor(np.zeros(1, dtype='int64')).to(device)
    with torch.no_grad():
        embeddings = net(data.unsqueeze(0))
        out = margin(embeddings, label)
    pred = out.cpu().detach().numpy()
    if not np.round(np.sum(pred[0]), decimals=2) == 1.0 or (
            np.round(np.sum(pred[0]), decimals=2) == 1.0 and any(n < 0 for n in pred[0])):
        pred[0] = softmax(pred[0])
    return pred


if __name__ == '__main__':
    bs = list()
    for i in range(0, 3):
        a = np.random.random_integers(low=0, high=255, size=[64, 3])
        bs.append(a[np.newaxis, :])
    c = np.concatenate(bs, axis=0)
    c = c.flatten()
    rangebound(c, [100, 0, 0], [255, 224 - 32, 224 - 32])
    print("-----")
