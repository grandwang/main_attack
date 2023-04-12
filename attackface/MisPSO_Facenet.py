import os.path
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

parser = argparse.ArgumentParser(description='MMPSO Parameters')
parser.add_argument('--dataset', '-d', type=str, help='Supports casia-webface, vggface2',
                    default='casia-webface')
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


def save_all_file_path(init_file_path, keyword):
    for cur_dir, sub_dir, included_file in os.walk(init_file_path):
        if included_file:
            for file in included_file:
                if re.search(keyword, file):
                    path.append(cur_dir + "\\" + file)
                    filename.append(file.split('.')[0])


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
    xp, yp = ORB_main(img, N)
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


def initX(net, xs, input, classes, watermark, sl, label):
    alpha = 0.99
    beta = 1 - alpha
    img = input.convert('RGB')
    fstpre, fstconfi = ec_main(net, img, '', classes)

    data_transform = transforms.Compose([
        transforms.Resize(112),
        transforms.ToTensor(), ])
    imglist = []
    # xs:(150,1)  递增i=i+3
    for i in range(0, len(xs)):
        Que = xs[i]
        attack_image = add_watermark_to_image(img, Que, watermark, sl).convert('RGB')
        attack_image_ele = data_transform(attack_image)
        imglist.append(attack_image_ele)

    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    label = torch.tensor(np.zeros(len(xs), dtype='int64')).to(device)
    batchimage_ele = torch.stack(imglist, dim=0)
    with torch.no_grad():
        embeddings = net(batchimage_ele.to(DEVICE))
        output = margin(embeddings, label).cpu()
        _, index = torch.min(output, 1)
        '''
        output:(50,1000)的值，需要找到 每个粒子 的fstconfi 对应分类 下的值，然后比较出一个最小
        1000类本身是按照顺序来的，找到一个也就是找到了其他49个粒子该分类对应的位置
        tensor --> numpy，按列索引，get到最小值，即对应xs的index，返回该xs
        ↑ 此思路仅用于第一次初始化解
        '''
        low_con = output.numpy()
        low_con = low_con[:, j]
        low_con_idx = np.argmin(low_con, axis=0)
        # 找出最小的atkconfi对应的xs值
        minxs = xs[low_con_idx]
        cost = low_con[low_con_idx]
    return cost, minxs


def init_position_x(X1, X2, input, label, net):  # (solu_plc_rand,solu_plc_slice,label,targetLabel,model)
    fit = np.zeros(1, dtype='float')
    fitP = float('inf') * np.ones(1, dtype='float')
    for k in range(1, 3):
        if k == 1:
            fit, X_best = initX(net, X1, input, classes,
                                watermark, sl, label)
        if k == 2:
            fit, X_best = initX(net, X2, input, classes,
                                watermark, sl, label)
        # choose fi || fnew
        if fit < fitP:
            X = np.full((numOfParticles, 3), X_best)
            fitP = fit
    return X


# x应为Swarm的坐标与alpha值，换言之这个方法应在初始化Swarm之后
def getBoundaries(x):
    # -1.0 与 1.0是 float32类型下的Boundary，不是tensor类型下的
    # rangebound in utilitie.py
    lowerBoundary = value_down_range
    upperBoundary = value_up_range
    return x, lowerBoundary, upperBoundary


def getModel_X(model_name):
    if dataset == 'casia-webface':
        model, input_size = initialize_model(model_name, num_classes=10575, feature_extract=True, use_pretrained=True)
    if dataset == 'vggface2':
        model, input_size = initialize_model(model_name, num_classes=8631, feature_extract=True, use_pretrained=False)
    return model, input_size


def createPlotPath(i, j):
    dirPath = os.path.join(plotDirs, str(i))
    if not os.path.isdir(dirPath):
        os.mkdir(dirPath)
    return dirPath


def pseudorandom_target(index, total_indices, true_class):
    rng = np.random.RandomState(index)
    target = true_class
    while target == true_class:
        target = rng.randint(0, total_indices)
    return target


# 控制定向与非定向
def getTargetLabel(pred):
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
    targetLabel = getTargetLabel(pred)
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
    swarm = Swarm(numOfParticles, model, targetLabel, maxChange, dataset, dLength, verbosity, topN, targeted, queries,
                  numofMethods, input, watermark, sl, input_size)
    pred = swarm.returnTopNPred(pred[0])
    print('Model Prediction Before PSO= %s' % (pred))
    init_flag, init_idx = fitnessScore('', model, solu_plc.reshape(numOfParticles * numofMethods, 3), input, classes,
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
    if verbosity >= 1:
        print('Baseline Confidence= %s' % (str(baselineConfidence)))
        # print('Baseline Fitness= %s\n'%(str(initialFitness)))
    swarm.initializeSwarmAndParticles(solu_plc, initialFitness, model, input, classes, j)
    return swarm, baselineConfidence, pred, targetLabel, dirPath, solu_plc


# input, N, dim, classes[j], path[i], filename[i],j

def mmpsoattack(input, N, dim, label, path, i, j):  # N, dim, label, path, filename-->x,y,g,i (testData,testLabels,g,count)

    pred = object_function(model, input)
    input, lowerBoundary, upperBoundary = getBoundaries(
        input)  # lowerBoundary & upperBoundary 作用于坐标每次更新的浮动，而不是adv-logo在全局的位置限制
    print("Searching Advresarial Example for test sample %s..." % (i))
    numberOfQueries = 0  # 单次迭代t下的目前的查询次数
    # 初始化粒子群初始解
    swarm, baselineConfidence, pred, targetLabel, dirPath, solu_plc = Initialization(pred, input, model, i,
                                                                                     lowerBoundary, upperBoundary, j,
                                                                                     classes, path, label)
    input.save(os.path.join(dirPath, 'Source.png'))
    if solu_plc == '':
        return 'skip'
    if baselineConfidence == '':
        AfterAtkImage = re2Image(solu_plc, dataset, input, watermark, sl)
        AfterAtkImage.save(os.path.join(dirPath, 'AfterPSO_arcface_mo.png'))
        return True
    # pred=swarm.returnTopNPred(pred[0])
    _, _, iterations, numberOfQueries, atkflag = swarm.searchOptimum(model, input, classes, j)
    finalFitness = swarm.bestFitness
    if atkflag:
        print("atk success:", label, " Now Label:", swarm.swarmLabel)
        AfterAtkImage = re2Image(swarm.swarmBestPosition, dataset, input, watermark, sl)
        AfterAtkImage.save(os.path.join(dirPath, 'AfterPSO_arcface_mo.png'))
        # g.show(AfterAtkImage, swarm.swarmLabel, save=True, path=os.path.join(dirPath, 'AfterPSO.png'))
        swarm.cleanSwarm()
        return True
    result_flag = False
    swarm.cleanSwarm()
    return result_flag


if __name__ == '__main__':
    random.seed(25)
    """
    采用新的搜索过程
    """
    count = 0
    count1 = 0
    dim = 3
    sl = 4  # 放缩系数
    # 处理logo图片
    '''[FaceNet, CosFace50, ArcFace50,MobileFaceNet] '''
    model_name = 'ArcFace50'
    model, input_size = getModel_X(model_name)
    watermark_logo = 'trans_patch'
    watermark, watermark_x1, watermark_y1,size = Open_logo(
        os.path.join('C:\\Codes\\demos\\logo\\','transfer_patch_resnet50_target.png'),sl,dataset)
    # watermark, watermark_x1, watermark_y1, size = Open_logo(
    #     os.path.join('E:\\PaperLearning\\PyProject\\Adv-watermark\\demos\\logo\\', 'Cambridge.png'), sl, dataset)

    # logo在图片中的贴图范围上下界，非变动范围
    value_up_range = [200, input_size - watermark_x1, input_size - watermark_y1]
    value_down_range = [100, 0, 0]

    # data_dir = r'C:\Datasets\casia-112x112\casia-112x112'
    data_dir = r'C:\Datasets\lfw-112x112\lfw_test'
    # prepareLogFilesAndOutputDirectoriy()

    transform = transforms.Compose([
        transforms.Resize(112),  # resize shortest side
        transforms.ToTensor()
    ])

    classes = sorted(os.listdir(data_dir))

    path = []
    filename = []
    save_all_file_path(data_dir, ".jpg")

    sampleList = sorted(random.sample(range(len(path)), 300))
    print('Attack list --> ', sampleList)
    for i in sampleList:  # Num of Images
        for j in range(len(classes)):
            matchObj = re.search(classes[j], path[i])
            if matchObj:
                pred = getpred(path[i], '', model, classes)
                if (pred == classes[j]):
                    start = time.time()
                    count = count + 1
                    # print('count：', count)
                    input = Image.open(path[i])
                    flag = mmpsoattack(input, numOfParticles, dim, classes[j], path[i], i, j)
                    if flag == True:
                        print(filename[i], "的攻击完成")
                        count1 += 1
                        print('共', count, "条数据, 成功率:", float(count1 / count), end='\n')
                    elif flag == False:
                        print(filename[i], "未能攻击成功", end='\n')
                        print('共', count, "条数据, 成功率:", float(count1 / count), end='\n')
                    else:
                        print('Skip')
                        count = count - 1
                    end = time.time()
                    print('Time:{}ms'.format((end - start) * 1000))
                else:
                    print("Model has orierror on this sample, skip", i, end='\n')
