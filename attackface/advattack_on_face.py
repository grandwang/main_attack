# -*- encoding:utf8 -*-"

import os
import torch
import torch.nn as nn
import numpy as np
import time
import joblib
from torchvision import datasets, transforms

from torchvision.datasets import ImageFolder
import re, skimage, time, random
import cv2
import argparse
from Dataset import *
from Utilities_face import *
from Swarm import Swarm
from particle import particle
import torch.nn.functional
import matplotlib.pyplot as plt
import sys

sys.path.append("../")
# from Loss_Modifier import NPSCalculator,TotalVariation
import random

parser = argparse.ArgumentParser(description='MISPSO Parameters')
parser.add_argument('--dataset', '-d', type=str, help='Supports casia-webface, vggface2, lfw',
                    default='lfw')
parser.add_argument('--maxChange', type=float,
                    help='Controls the L-infinity distance between the source and destination images', default=20)
parser.add_argument('--numOfParticles', '-p', type=int, help='Number of particles in the swarm', default=40)
parser.add_argument('--blockSize', type=int, help='Initial blocksize for seperating image into tiles', default=4)
parser.add_argument('--targeted', '-t', help='Choose random target when crafting examples', default=False,
                    action='store_true')
parser.add_argument('--C1', type=float, help='Controls exploitation weight', default=2.0)
parser.add_argument('--C2', type=float, help='Controls explorations weight', default=2.0)
parser.add_argument('--image_size', type=int, help='Input Image size,suit with model', default=112)

parser.add_argument('--Samples', '-n', type=int, help='Number of test Samples to attack', default=10000)
parser.add_argument('--Randomize', help='Randomize dataset', default=False, action='store_true')
parser.add_argument('--verbose', '-v', type=int,
                    help='Verbosity level. 0 for no terminal logging, 1 for samples results only, and 2 for swarm level verbosity',
                    default=2)
parser.add_argument('--topN', type=int, help='Specify the number of labels to reduce when attacking imagenet',
                    default=1)
parser.add_argument('--sample', type=int, help='Specify which sample to attack', default=-1)
parser.add_argument('--Queries', '-q', type=int, help='Mazimum number of queries', default=5)
parser.add_argument('--pars', help='Run in Parsimonious... samples', default=False, action='store_true')
parser.add_argument('--numofMethods', type=int, help='How many kinds of Image processing algorithm you use', default=3)

args = parser.parse_args()

maxChange = args.maxChange  # 攻击前后图片允许的最大差异
numOfParticles = args.numOfParticles  # 集群最大粒子数
targeted = args.targeted  # 指向性or非指向性攻击
N = args.Samples  # 本次运行要攻击的图片数量
verbosity = args.verbose  # 记录详细程度
C1 = args.C1  # 局部学习率
C2 = args.C2  # 全局学习率
Randomize = args.Randomize  # 坐标变化率
dataset = args.dataset  # 数据集类型
sample = args.sample  # 是否指定攻击某个样本
blockSize = args.blockSize  # 用于将图像分块的初始块大小, 8x8
queries = args.Queries  # 最大查询次数，数量
pars = args.pars
numofMethods = args.numofMethods + 1  # 使用多少种图像处理方法去产生初始解,+1是因为还有一个比较的出的解

# 保存数据路径——审稿意见修改用
watermark_logo = 'transfer_patch'  # one more in mid_save()
reviewer_path = os.path.join(watermark_logo, "C:/Codes/main_attack/attackface/Results"
                                             "/reviewer#3/comment#6/") + watermark_logo

if not 'imagenet' in dataset and args.topN > 1:
    topN = 1
    print('Top N only supports attacks on imagenet. Resetting to 1\n')
else:
    topN = args.topN
if dataset == 'casia-webface':
    dLength = numofMethods * numOfParticles * 3
    plotDirs = os.path.join('.', 'Results', 'casia-webface')
elif dataset == 'vggface2':
    dLength = numofMethods * numOfParticles * 3
    plotDirs = os.path.join('.', 'Results', 'vggface2')
elif dataset == 'lfw':
    dLength = numofMethods * numOfParticles * 3
    plotDirs = os.path.join('.', 'Results', 'lfw')

correctlyClassified = 0


def prepareLogFilesAndOutputDirectoriy():
    if not os.path.isdir(os.path.join('.', 'Results')):
        os.mkdir(os.path.join('.', 'Results'))
    if not os.path.isdir(plotDirs):
        os.mkdir(plotDirs)
    with open(os.path.join('.', 'Results', dataset + '_PSO_Results.csv'), 'w') as f:
        f.write(
            'Sample,BaselineCofidence,BaselineFitness,TargetLabel,Prediction_Before_PSO, Confidence_After_PSO,Fitness_After_PSO,Prediction_After_PSO,Iteration,L2_Difference_Between_Images,L0,LInfinity,Number_of_Model_Queries,Results\n')


# def save_all_file_path(init_file_path, keyword):
#     for cur_dir, sub_dir, included_file in os.walk(init_file_path):
#         if included_file:
#             for file in included_file:
#                 if re.search(keyword, file):
#                     path.append(cur_dir + "\\" + file)
#                     filename.append(file.split('.')[0])


'''
全局优化算法所需部分:控制边界
'''


def init_position_select_range(x):
    if x > 112 - watermark_x1:
        return 112 - watermark_x1
    else:
        return x


from SURF import main as SURF_main
from ORB import main as ORB_main
from Dlib import main as Dlib_main


def init_position_SURF(N, dim, path):
    X1 = []
    img = cv2.imread(path)
    img = cv2.resize(img, (112, 112))
    xp, yp = SURF_main(img, N)
    for i in range(0, N):
        X1list = []
        for j in range(dim):
            if j == 0:
                X1list.append(100 + random.random() * (255 - 100))
            if j == 1:
                X1list.append(init_position_select_range(xp[i]))
            if j == 2:
                X1list.append(init_position_select_range(yp[i]))
        X1.append(X1list)
    return np.array(X1)


def init_position_ORB(N, dim, path):
    X1 = []
    img = cv2.imread(path)
    img = cv2.resize(img, (112, 112))
    xp, yp = ORB_main(img, N, input_size)
    for i in range(0, N):
        X1list = []
        for j in range(dim):
            if j == 0:
                X1list.append(100 + random.random() * (255 - 100))
            if j == 1:
                X1list.append(init_position_select_range(xp[i]))
            if j == 2:
                X1list.append(init_position_select_range(yp[i]))
        X1.append(X1list)
    return np.array(X1)


def init_position_Dlib(N, dim, path):
    X1 = []
    img = cv2.imread(path)
    img = cv2.resize(img, (112, 112))
    xp, yp = Dlib_main(img, N)
    for i in range(0, N):
        X1list = []
        for j in range(dim):
            if j == 0:
                X1list.append(100 + random.random() * (255 - 100))
            if j == 1:
                X1list.append(init_position_select_range(xp[i]))
            if j == 2:
                X1list.append(init_position_select_range(yp[i]))
        X1.append(X1list)
    return np.array(X1)


def init_position_random(N, dim):
    X1 = []
    for i in range(0, N):
        X1list = []
        for j in range(dim):
            X1list.append(value_down_range[j] + random.random() * (value_up_range[j] - value_down_range[j]))
        X1.append(X1list)
    # print("random: ", X1)
    X1 = np.array(X1)
    return X1


# def initX(model, xs, input, classes, watermark, sl, label):
#     alpha = 0.99
#     beta = 1 - alpha
#     img = input.convert('RGB')
#     fstpre, fstconfi = cos_main(model, img, '', classes)
#
#     data_transform = transforms.Compose([
#         transforms.Resize(112),
#         transforms.ToTensor(), ])
#     imglist = []
#     # xs:(150,1)  递增i=i+3
#     for i in range(0, len(xs)):
#         Que = xs[i]
#         attack_image = add_watermark_to_image(img, Que, watermark, sl).convert('RGB')
#         attack_image_ele = data_transform(attack_image)
#         imglist.append(attack_image_ele)
#
#     DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     label = torch.tensor(np.zeros(len(xs), dtype='int64')).to(device)
#     batchimage_ele = torch.stack(imglist, dim=0)
#     with torch.no_grad():
#         embeddings = model(batchimage_ele.to(DEVICE))
#         output = margin(embeddings, label).cpu()
#         _, index = torch.min(output, 1)
#         '''
#         output:(50,1000)的值，需要找到 每个粒子 的fstconfi 对应分类 下的值，然后比较出一个最小
#         1000类本身是按照顺序来的，找到一个也就是找到了其他49个粒子该分类对应的位置
#         tensor --> numpy，按列索引，get到最小值，即对应xs的index，返回该xs
#         ↑ 此思路仅用于第一次初始化解
#         '''
#         low_con = output.numpy()
#         low_con = low_con[:, j]
#         low_con_idx = np.argmin(low_con, axis=0)
#         # 找出最小的atkconfi对应的xs值
#         minxs = xs[low_con_idx]
#         cost = low_con[low_con_idx]
#     return cost, minxs

def initX(model, xs, input, classes, watermark, sl, label):
    alpha = 0.99
    beta = 1 - alpha
    img = input.convert('RGB')
    # fstpre, fstconfi = cos_main(model, img, '', classes)
    shutil.rmtree('./initX')
    data_transform = transforms.Compose([
        transforms.Resize(112),
        transforms.ToTensor()])
    imglist = []
    if not os.path.exists("initX/{}".format(label)):
        os.makedirs("initX/{}".format(label))
    for i in range(0, len(xs)):
        Que = xs[i]
        attack_image = add_watermark_to_image(img, Que, watermark, sl).convert('RGB')
        attack_image.save('initX/{}/{}.png'.format(label, i))
        attack_image_ele = data_transform(attack_image)
        imglist.append(attack_image_ele)

    init_data = datasets.ImageFolder('initX')
    batchimage_ele=[]
    for i in range(len(init_data)):
        batchimage_ele.append(init_data[i][0])
    # batchimage_ele = torch.stack(imglist, dim=0)
    with torch.no_grad():
        # crops_result, crops_tensor = crop_imgs(batchimage_ele.cpu().numpy(), inputsize[model_name][0],
        #                                        inputsize[model_name][1])
        out_result, out_tensor = crop_imgs(batchimage_ele, inputsize[model_name][0],
                                               inputsize[model_name][1])
        # embeddings = model(batchimage_ele.to(DEVICE))
        sim_labels, sim_probs = check_all(out_tensor, model, model_name, device)
        _, index = torch.min(torch.tensor(sim_probs), 1)
        _, indices = torch.sort(torch.tensor(sim_probs), descending=False)
        # low_con = output.numpy()
        low_con = sim_probs
        # low_con = low_con[:, start_label[0]]
        low_con = low_con[:, start_label[0]]
        low_con_idx = np.argmin(low_con, axis=0)
        # 找出最小的atkconfi对应的xs值
        minxs = xs[low_con_idx]
        cost = low_con[low_con_idx]
    del i
    shutil.rmtree('./initX')
    os.mkdir('./initX')
    return cost, minxs


def init_position_x(X1, X2, input, label, net):  # (solu_plc_rand,solu_plc_slice,label,targetLabel,model)
    fit = np.zeros(1, dtype='float')
    fitP = float('inf') * np.ones(1, dtype='float')
    for k in range(1, 3):
        if k == 1:
            fit, X_best = initX(net, X1, input, lfw_dataset.classes,
                                watermark, sl, label)
        if k == 2:
            fit, X_best = initX(net, X2, input, lfw_dataset.classes,
                                watermark, sl, label)
        # choose fi || fnew
        if fit < fitP:
            X = np.full((numOfParticles, 3), X_best)
            fitP = fit
    return X


# x应为Swarm的坐标与alpha值，换言之这个方法应在初始化Swarm之后
def getBoundaries(input):
    # -1.0 与 1.0是 float32类型下的Boundary，不是tensor类型下的
    # rangebound in utilitie.py
    lowerBoundary = value_down_range
    upperBoundary = value_up_range
    # input = transform(input)
    return input, lowerBoundary, upperBoundary


def getModel_X(model_name):
    if dataset == 'casia-webface' or dataset == 'lfw':
        model, input_size = initialize_model(model_name, num_classes=10575, feature_extract=True, use_pretrained=True)
    if dataset == 'vggface2':
        model, input_size = initialize_model(model_name, num_classes=8631, feature_extract=True, use_pretrained=False)
    return model, input_size


def createPlotPath(i, j):
    dirPath = os.path.join(plotDirs, str(i))
    if not os.path.isdir(dirPath):
        os.makedirs(dirPath)
    return dirPath


def pseudorandom_target(index, total_indices, true_class):
    rng = np.random.RandomState(index)
    target = true_class
    while target == true_class:
        target = rng.randint(0, total_indices)
    return target


# 控制定向与非定向
def getTargetLabel(pred):
    # if targeted == True and topN == 1:
    #     targetLabel = pseudorandom_target(i, len(pred[0]), np.argmax(pred))
    # elif targeted == True and topN > 1:
    #     targetLabel = list(set(g.getClassName().keys()).difference(np.argsort(pred[0])[:][::-1][:topN]))
    #     targetLabel = [random.choice(targetLabel)]
    #     targetLabel.extend(np.argsort(pred[0])[:][::-1][:topN - 1])
    # else:
    #     if topN > 1:
    #         targetLabel = np.argsort(pred[0])[:][::-1][:topN]
    #     elif topN == 1:
    targetLabel = np.argmax(pred)
    return targetLabel


def Initialization(pred, input, model, i, lowerBoundary, upperBoundary, j, classes, path, label):
    '''
    pred,—————— 模型正确预测结果,int64
    input,—————— 要攻击的底图
    model, —————— 要攻击的模型
    i,j, ———— 第i张底图，第j张
    lowerBoundary,upperBoundary,———— 变动上下界
    classes,—————— label集合
    solu_plc == solution_particles_random
    '''
    dirPath = createPlotPath(i, j)
    # targetLabel = getTargetLabel(pred)
    targetLabel = start_label.copy()
    # 初始化粒子群
    # x:[100,0,0] x[0]:alpha  x[1]:x_position x[2]:y_position i+2
    '''
    ori method: rand, surf, orb
    pro method: rand, dlib, orb
    '''
    solu_plc_rand = init_position_random(numOfParticles, dim)
    solu_plc_slice = init_position_ORB(numOfParticles, dim, path)
    # solu_plc_temp = init_position_SURF(numOfParticles, dim, path)
    # solu_plc_slice = init_position_random(numOfParticles, dim)
    # solu_plc_temp = init_position_random(numOfParticles, dim)
    solu_plc_face = init_position_Dlib(numOfParticles, dim, path)
    solu_plc = init_position_x(solu_plc_slice, solu_plc_face, input, label, model)

    # 暂时把solu_plc放在首位方便计算baselineConfidence
    solu_plc = np.array([solu_plc, solu_plc_rand, solu_plc_slice, solu_plc_face])
    solu_plc = solu_plc.flatten()
    # swarm = Swarm(numOfParticles, model, targetLabel, maxChange, dataset, dLength, verbosity, topN, targeted, queries,
    #               numofMethods, input, watermark, sl, input_size)
    swarm = Swarm(numOfParticles, model, label, maxChange, dataset, dLength, verbosity, topN, targeted, queries,
                  numofMethods, input, watermark, sl, input_size)
    pred = swarm.returnTopNPred(pred[0])
    print('Model Prediction Before PSO= %s' % (pred))
    init_flag, init_idx = fitnessScore('', model, solu_plc.reshape(numOfParticles * numofMethods, 3), input, label,
                                       watermark, sl, j)
    if init_flag == False:
        print(init_idx)
        return '', '', '', '', '', ''
    if init_flag == True:
        print("atk success:", label, " Now Label:", init_idx[0])
        return '', '', '', '', dirPath, init_idx[1]

    # g.show(input, pred, save=True, path=os.path.join(dirPath, 'Original.png'))
    # ori_pre == baselineConfidence     #baselineConfidence=swarm.initX(model,solu_plc,targetLabel,input,classes,watermark,sl)
    baselineConfidence = swarm.calculateBaselineConfidence(model, targetLabel, input)

    swarm.setSwarmAttributes(solu_plc, C1, C2, lowerBoundary, upperBoundary, blockSize)
    initialFitness = np.ones(numofMethods * numOfParticles, dtype='float')
    swarm.setInitialFitness(initialFitness)
    # if verbosity >= 1:
        # print('Baseline Confidence= %s' % (str(baselineConfidence)))
            # print('Baseline Fitness= %s\n'%(str(initialFitness)))
    swarm.initializeSwarmAndParticles(solu_plc, initialFitness, model, input, classes, j)
    return swarm, baselineConfidence, pred, targetLabel, dirPath, solu_plc


# input, N, dim, classes[j], path[i], filename[i],j

def mmpsoattack(input, N, dim, label, path, i,
                j):  # N, dim, label, path, filename-->x,y,g,i (testData,testLabels,g,count)

    # pred = object_function(model, input)
    pred = sim_probs.copy()
    if not np.round(np.sum(pred[0]), decimals=2) == 1.0 or (
            np.round(np.sum(pred[0]), decimals=2) == 1.0 and any(n < 0 for n in pred[0])):
        pred[0] = softmax(pred[0])

    input, lowerBoundary, upperBoundary = getBoundaries(
        input)  # lowerBoundary & upperBoundary 作用于坐标每次更新的浮动，而不是adv-logo在全局的位置限制
    print("Searching Advresarial Example for test sample %s..." % (i))
    numberOfQueries = 0  # 单次迭代t下的目前的查询次数
    # 初始化粒子群初始解
    swarm, baselineConfidence, pred, targetLabel, dirPath, solu_plc = Initialization(pred, input, model, i,
                                                                                     lowerBoundary, upperBoundary, truelabel,
                                                                                     lfw_dataset.classes, path, label) #j
    input.save(os.path.join(dirPath, 'Source.png'))
    if solu_plc == '':
        return 'skip'
    if baselineConfidence == '':
        # print("atk success:", label, " Now Label:", swarm.swarmLabel)
        # AfterAtkImage = re2Image(solu_plc, dataset, input, watermark, sl)
        # AfterAtkImage.save(os.path.join(dirPath, 'AfterPSO_arcface_mo.png'))
        return True
    # pred=swarm.returnTopNPred(pred[0])
    _, _, iterations, numberOfQueries, atkflag = swarm.searchOptimum(model, input, lfw_dataset.classes, truelabel) #j
    finalFitness = swarm.bestFitness
    if atkflag:
        print("atk success:", label, " Now Label:", swarm.swarmLabel)
        # AfterAtkImage = re2Image(swarm.swarmBestPosition, dataset, input, watermark, sl)
        # AfterAtkImage.save(os.path.join(dirPath, 'AfterPSO_arcface_mo.png'))
        # g.show(AfterAtkImage, swarm.swarmLabel, save=True, path=os.path.join(dirPath, 'AfterPSO.png'))
        swarm.cleanSwarm()
        return True
    result_flag = False
    swarm.cleanSwarm()
    return result_flag


trans = transforms.Compose([
    transforms.Resize(112),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
])


def t2image(tensor):
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()  # clone the tensor
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    image.save('example.jpg')


def crop_imgs(imgs, w, h):
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
    embedding_sets = joblib.load('C:/Codes/demos/reviewer#4/comment6/newpatch-rl-new/rlpatch/'
                                 'stmodels/{}/embeddings_{}.pkl'.format(model_name, model_name))
    sets = torch.t(embedding_sets).to(device)  # (512,5749)
    # print(embedding.shape,sets.shape)
    numerator = torch.mm(feature, sets)
    norm_x1 = torch.norm(feature, dim=1)
    norm_x1 = torch.unsqueeze(norm_x1, 1)
    norm_x2 = torch.norm(sets, dim=0)  # ,keepdims=True
    norm_x2 = torch.unsqueeze(norm_x2, 0)
    # print('norm_x1,norm_x2 ',norm_x1.shape,norm_x2.shape)
    denominator = torch.mm(norm_x1, norm_x2)
    metrics = torch.mul(numerator, 1 / denominator)
    return metrics.cpu().detach()


def check_all(adv_face_ts, threat, threat_name, device):
    # adv_tensor = [trans(adv_face)]
    # advface_ts = torch.stack(adv_tensor).to(device)
    percent = []
    typess = []

    # #print(adv_face_ts)
    # adv_face_arr = np.uint8(adv_face_ts.numpy()*255)
    # #print(adv_face_arr)
    # adv_face_ts = torch.from_numpy(adv_face_arr/255).half()
    # #print(adv_face_ts)

    def collate_fn(x):
        return x

    loader = DataLoader(
        adv_face_ts,
        batch_size=55,
        shuffle=False,
        collate_fn=collate_fn
    )

    for X in loader:
        # print(X[0].shape)
        advface_ts = torch.stack(X).to(device)  # list tensor{3,112,112}
        X_op = nn.functional.interpolate(advface_ts, (inputsize[threat_name][0], inputsize[threat_name][1]),
                                         mode='bilinear', align_corners=False)
        feature = threat(X_op)  # tensor (1,512)
        for i in range(len(feature)):
            sim_all = cosin_all(torch.unsqueeze(feature[i], 0), threat_name, device)
            _, indices = torch.sort(sim_all, dim=1, descending=True)
            cla = [indices[0][0].item(), indices[0][1].item(), indices[0][2].item(),
                   indices[0][3].item(), indices[0][4].item(), indices[0][5].item(), indices[0][6].item()]
            typess.append(cla)
            tage = sim_all[0].numpy()
            percent.append(tage)
    return typess, np.array(percent)


def attack_process(img, label, threat_model, threat_name, device, width, height):
    crops_result, crops_tensor = crop_imgs([img], width,
                                           height)
    # convert face image to tensor RL framework iterations

    sim_labels, sim_probs = check_all(crops_tensor, threat_model, threat_name, device)

    start_label = sim_labels[0][:2]
    start_gap = sim_probs[0][start_label]
    # target = sim_labels[0][1] if targeted else sim_labels[0][0]
    truelabel = sim_labels[0][0]
    print('start_label: {} start_gap: {}  truelabel:{}'.format(start_label, start_gap, label))


def draw_convergence(data, path):
    # 每隔5个点取一个点
    data_new = []
    for i in range(len(data)):
        if i % 5 == 0 and i != 0:
            data_new.append(data[i])
    plt.figure(figsize=(6, 6), dpi=300)
    x = list(np.arange(1, len(data_new) + 1))
    y = list(abs(np.array(data_new)))
    plt.plot(x, y, ls='-', c='m', linewidth=2, marker='o', markersize=4)
    plt.draw()
    plt.savefig(path + 'ASR_fig_{}_{}_{}.png'.format(watermark_logo, model_name, len(data)))
    print("Draw Over")


if __name__ == "__main__":
    random.seed(10)
    """
    采用新的搜索过程
    """
    count = 0
    count1 = 0
    dim = 3
    sl = 8  # 放缩系数
    # 处理logo图片
    '''[FaceNet, CosFace50, ArcFace50,MobileFaceNet] '''
    model_name = 'arcface50'
    model, input_size = getModel_X(model_name)
    watermark_logo = 'trans_patch'
    watermark, watermark_x1, watermark_y1, size = Open_logo(
        # os.path.join('C:\\Codes\\demos\\logo\\', 'transfer_patch_resnet50_target.png'), sl, dataset)
        os.path.join('C:\\Codes\\demos\\logo\\', 'best_patch.png'), sl, dataset)
    # watermark, watermark_x1, watermark_y1, size = Open_logo(
    #     os.path.join('E:\\PaperLearning\\PyProject\\Adv-watermark\\demos\\logo\\', 'Cambridge.png'), sl, dataset)

    # logo在图片中的贴图范围上下界，非变动范围
    value_up_range = [200, input_size - watermark_x1, input_size - watermark_y1]
    value_down_range = [100, 0, 0]

    # data_dir = r'C:\Datasets\casia-112x112\casia-112x112'
    data_dir = r'C:\Datasets\lfw-112x112\lfw-112x112'

    transform = transforms.Compose([
        transforms.Resize(112),  # resize shortest side
        # transforms.ToTensor()
    ])

    # classes = sorted(os.listdir(data_dir))
    #
    # path = []
    # filename = []
    # save_all_file_path(data_dir, ".jpg")
    #
    # sampleList = sorted(random.sample(range(len(path)), 300))
    # print('Attack list --> ', sampleList)

    '''-------------------------------------------------------------------------------------------'''
    inputsize = {'arcface34': [112, 112], 'arcface50': [112, 112], 'cosface34': [112, 112], 'cosface50': [112, 112],
                 'facenet': [160, 160], 'insightface': [112, 112], 'sphere20a': [112, 96], 're_sphere20a': [112, 96],
                 'mobilefacenet': [112, 112], 'tencent': [112, 112]}

    localtime1 = time.asctime(time.localtime(time.time()))
    localtime1 = localtime1.replace(":", "_").replace(" ", "_")
    lfw_dataset = datasets.ImageFolder(data_dir)
    sampleList = sorted(random.sample(range(len(lfw_dataset.samples)), 500))
    print(sampleList)
    lfw_dataset.idx_to_class = {i: c for c, i in lfw_dataset.class_to_idx.items()}
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    start_all = time.perf_counter()
    asr_t1 = 0
    asr_t2 = 0
    ir_flag = True
    asr = []
    mid_flag = True

    for i in range(len(lfw_dataset)):
        idx = i
        # ori_label refers to the ori label in Class "dataset"
        ori_data = lfw_dataset[sampleList[idx]][0]  # input
        ori_label = lfw_dataset[sampleList[idx]][1]  # j
        path = lfw_dataset.imgs[sampleList[idx]][0]  # path[i]
        classname = lfw_dataset.classes[sampleList[idx]]  # label & classes[i]

        crops_result, crops_tensor = crop_imgs([ori_data], inputsize[model_name][0],
                                               inputsize[model_name][1])
        # convert face image to tensor RL framework iterations
        sim_labels, sim_probs = check_all(crops_tensor, model, model_name, device)

        # start_label: model given
        start_label = sim_labels[0][:2]
        start_pre = sim_probs[0][start_label]
        # target = sim_labels[0][1] if targeted else sim_labels[0][0]
        truelabel = sim_labels[0][0]
        print('start_label: {} start_gap: {}  truelabel:{}'.format(start_label, start_pre, truelabel))

        start = time.perf_counter()
        count = count + 1
        # print('count：', count)
        # input = Image.open(path[i])
        if (mmpsoattack(ori_data, numOfParticles, dim, truelabel, path, idx, ori_label)):
            print(classname, "的攻击完成")
            count1 += 1
            print('共', count, "条数据, 成功率:", float(count1 / count), end='\n')
            # 实验终止条件
            asr_t2 = float(count1 / count)
            ir_bound = abs(asr_t2 - asr_t1)
            asr.append(asr_t2)  # 收集数据绘图
            if ir_bound >= 0.001 and count > 300:
                print("趋于收敛")
                ir_flag = False
                break
            asr_t1 = asr_t2
        else:
            print(classname, "未能攻击成功", end='\n')
            print('共', count, "条数据, 成功率:", float(count1 / count), end='\n')
            # 实验终止条件
            asr_t2 = float(count1 / count)
            ir_bound = abs(asr_t2 - asr_t1)
            asr.append(asr_t2)  # 收集数据绘图
            if ir_bound >= 0.001 and count > 300:
                print("趋于收敛")
                ir_flag = False
                break
            asr_t1 = asr_t2
        end = time.perf_counter()
        print('Time:{}s'.format((end - start)))

        if count >= 150 and mid_flag is True:
            draw_convergence(asr, reviewer_path)
            mid_flag = False
        if not ir_flag:
            break
    end_all = time.perf_counter()
    print('All_Time:{}s'.format((end_all - start_all)))
    draw_convergence(asr, reviewer_path)
