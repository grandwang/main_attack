import numpy as np
from numpy.random import rand
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from torch import nn
import os
from PIL import Image, ImageDraw


def init_p_position(lb, ub, N, dim):
    # p&q_quo use to calculate eq36&eq37 as TVSM-BPSO
    P1 = np.zeros([N, dim], dtype='float')
    for i in range(N):
        for d in range(dim):
            P1[i, d] = lb[0, d] + (ub[0, d] - lb[0, d]) * rand()
    P = P1
    P2 = P1
    P1_quo = np.zeros([N, dim], dtype='float')
    for i in range(N):
        for d in range(dim):
            P1_quo[i, d] = lb[0, d] + (ub[0, d] - lb[0, d]) * rand()
    P_quo = P1_quo
    P2_quo = P1_quo
    return P, P_quo, P1, P1_quo, P2, P2_quo


def init_velocity(lb, ub, N, dim):
    V1 = np.zeros([N, dim], dtype='float')
    Vmax = np.zeros([1, dim], dtype='float')
    Vmin = np.zeros([1, dim], dtype='float')
    for d in range(dim):
        Vmax[0, d] = (ub[0, d] - lb[0, d]) / 2
        Vmin[0, d] = -Vmax[0, d]
    for i in range(N):
        for d in range(dim):
            V1[i, d] = Vmin[0, d] + (Vmax[0, d] - Vmin[0, d]) * rand()
    V = V1
    V2 = V1
    return V, V1, V2, Vmax, Vmin


def binary_conversion(X, thres, N, dim):
    Xbin = np.zeros([N, dim], dtype='int')
    for i in range(N):
        for d in range(dim):
            if X[i, d] > thres:
                Xbin[i, d] = 1
            else:
                Xbin[i, d] = 0
    return Xbin


def cal_sigema(sigema_max, sigema_min, t, max_iter):
    sigema = (sigema_max - sigema_min) * (t / max_iter) + sigema_min
    return sigema


def sigfun(velocity, sigema):
    superscript = float(sigema * velocity)
    return 1 / (float(1 + np.exp(superscript)))


def sigfun_quo(velocity, sigema):
    superscript = float(sigema * (-velocity))
    return 1 / (float(1 + np.exp(superscript)))


def update_w(w_max, w_min, N, t):
    w = w_max - (w_max - w_min) * t / N
    return w


def chooseQ(x, y):
    if x.shape == y.shape:
        if np.sum(x) > np.sum(y):
            return x
        else:
            return y
    else:
        print("Error on dimentions")
    return


def boundary(x, Vmin, Vmax):
    if x < Vmin:
        x = Vmin
    if x > Vmax:
        x = Vmax
    return x


def add_watermark_to_image(image, xs, watermark, sl):
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
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
    a = np.clip(a, 0, 224 - watermark_x)
    b = np.array(xs[2])
    b = np.clip(b, 0, 224 - watermark_y)
    x_pos = int(a)
    y_pos = int(b)
    rgba_image.paste(rgba_watermark, (x_pos, y_pos), rgba_watermark_mask)
    # Image._show(rgba_image)
    return rgba_image


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
