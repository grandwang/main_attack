import os.path
from torchvision.datasets import ImageFolder
import re, skimage, time, random
import cv2
from tqdm import tqdm
import argparse

from Dataset import *
from Utilities import *
from Swarm import Swarm
from particle import particle
import torch.nn.functional
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='MMPSO Parameters')
parser.add_argument('--dataset', '-d', type=str, help='Supports CIFAR10, MNIST, imagenet224',
                    default='imagenet224')
parser.add_argument('--maxChange', type=float,
                    help='Controls the L-infinity distance between the source and destination images', default=30)
parser.add_argument('--numOfParticles', '-p', type=int, help='Number of particles in the swarm', default=20)
parser.add_argument('--targeted', '-t', help='Choose random target when crafting examples', default=False,
                    action='store_true')
parser.add_argument('--C1', type=float, help='Controls exploitation weight', default=2.0)
parser.add_argument('--C2', type=float, help='Controls explorations weight', default=2.0)
parser.add_argument('--Samples', '-n', type=int, help='Number of test Samples to attack', default=10000)
parser.add_argument('--Randomize', help='Randomize dataset', default=False, action='store_true')
parser.add_argument('--verbose', '-v', type=int,
                    help='Verbosity level. 0 for no terminal logging, 1 for samples results only, and 2 for swarm level verbosity',
                    default=2)
parser.add_argument('--topN', type=int, help='Specify the number of labels to reduce when attacking imagenet',
                    default=1)
parser.add_argument('--sample', type=int, help='Specify which sample to attack', default=-1)
parser.add_argument('--blockSize', type=int, help='Initial blocksize for seperating image into tiles', default=4)
parser.add_argument('--Queries', '-q', type=int, help='Mazimum number of queries', default=5)
parser.add_argument('--pars', help='Run in Parsimonious... samples', default=False, action='store_true')
parser.add_argument('--numofMethods', type=int, help='How many kinds of Image processing algorithm you use', default=3)

args = parser.parse_args()
maxChange = args.maxChange
numOfParticles = args.numOfParticles
targeted = args.targeted
N = args.Samples
verbosity = args.verbose
C1 = args.C1
C2 = args.C2
Randomize = args.Randomize
dataset = args.dataset
sample = args.sample
blockSize = args.blockSize
queries = args.Queries
pars = args.pars
numofMethods = args.numofMethods + 1

if not 'imagenet' in dataset and args.topN > 1:
    topN = 1
    print('Top N only supports attacks on imagenet. Resetting to 1\n')
else:
    topN = args.topN
if dataset == 'CIFAR10':
    dLength = 32 * 32 * 3
    plotDirs = os.path.join('.', 'Results', 'Plots_CIFAR')
elif dataset == 'MNIST':
    dLength = 28 * 28
    plotDirs = os.path.join('.', 'Results', 'Plots_MNIST')
elif dataset == 'imagenet224':
    dLength = numofMethods * numOfParticles * 3

    plotDirs = os.path.join('.', 'Results', 'Elsvier')

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
    if x > 224 - watermark_x1:
        return 224 - watermark_x1
    else:
        return x


from SURF import main as SURF_main
from ORB import main as ORB_main


def init_position_select(N, dim, path):
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


def init_position_random(N, dim):
    X1 = []
    for i in range(0, N):
        X1list = []
        for j in range(dim):
            X1list.append(value_down_range[j] + random.random() * (value_up_range[j] - value_down_range[j]))
        X1.append(X1list)

    X1 = np.array(X1)
    return X1


def initX(net, xs, input, classes, watermark, sl, label):
    alpha = 0.99
    beta = 1 - alpha
    img = input.convert('RGB')
    fstpre, fstconfi = ec_main(net, img, '', classes)

    data_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(), ])
    imglist = []

    for i in range(0, len(xs)):
        Que = xs[i]
        attack_image = add_watermark_to_image(img, Que, watermark, sl).convert('RGB')
        attack_image_ele = data_transform(attack_image)
        imglist.append(attack_image_ele)

    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batchimage_ele = torch.stack(imglist, dim=0)
    net.eval()
    with torch.no_grad():
        output = net(batchimage_ele.to(DEVICE)).cpu()
        _, index = torch.min(output, 1)
        percentage = torch.nn.functional.softmax(output, dim=1)[0]
        _, indices = torch.sort(output, descending=False)
        low_con = output.numpy()
        low_con = low_con[:, j]
        low_con_idx = np.argmin(low_con, axis=0)

        minxs = xs[low_con_idx]
        cost = low_con[low_con_idx]
    return cost, minxs


def init_position_x(X1, X2, input, label, net):
    i = 0
    X = init_position_random(numOfParticles, dim)
    fit = np.zeros(1, dtype='float')
    fitP = float('inf') * np.ones(1, dtype='float')
    for k in range(1, 3):
        if k == 1:
            fit, X_best = initX(net, X1, input, classes, watermark, sl, label)
        if k == 2:
            fit, X_best = initX(net, X2, input, classes, watermark, sl, label)

        if fit < fitP:
            X = np.full((numOfParticles, 3), X_best)
            fitP = fit
    return X


def getBoundaries(x):
    if dataset == 'CIFAR10' or dataset == 'MNIST':
        if np.min(x) < 0:
            lowerBoundary = -0.5
            upperBoundary = 0.5
        else:
            lowerBoundary = 0.0
            upperBoundary = 1.0
            x = np.add(x, 0.5)
    elif 'imagenet' in dataset:
        lowerBoundary = value_down_range
        upperBoundary = value_up_range
    return x, lowerBoundary, upperBoundary


def getModel_X(model_name):
    if dataset == 'imagenet224':
        model, input_size = initialize_model(model_name, num_classes=len(classes), feature_extract=True,
                                             use_pretrained=True)

    if dataset == 'CIFAR10':
        model, input_size = initialize_model(model_name, num_classes=10, feature_extract=True, use_pretrained=True)
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


def getTargetLabel(pred):
    if topN > 1:
        targetLabel = np.argsort(pred[0])[:][::-1][:topN]
    elif topN == 1:
        targetLabel = np.argmax(pred)
    return targetLabel


def Initialization(pred, input, model, i, lowerBoundary, upperBoundary, j, classes, path, label):
    dirPath = createPlotPath(i, j)

    targetLabel = getTargetLabel(pred)
    solu_plc_rand = init_position_random(numOfParticles, dim)

    solu_plc_slice = init_position_select(numOfParticles, dim, path)

    solu_plc_temp = init_position_ORB(numOfParticles, dim, path)

    solu_plc = init_position_x(solu_plc_rand, solu_plc_slice, input, label, model)

    solu_plc = np.array([solu_plc, solu_plc_rand, solu_plc_slice, solu_plc_temp])

    solu_plc = solu_plc.flatten()

    swarm = Swarm(numOfParticles, model, targetLabel, maxChange, dataset, dLength, verbosity, topN, targeted, queries,
                  numofMethods, input, watermark, sl, input_size)
    pred = swarm.returnTopNPred(pred[0])
    print('Model Prediction Before PSO= %s' % (pred))
    init_flag, init_idx = fitnessScore('', model, solu_plc.reshape(numOfParticles * numofMethods, 3), input, classes,
                                       watermark, sl, j)
    if init_flag == True:
        print("atk success:", label, " Now Label:", init_idx[0])
        return '', '', '', '', dirPath, init_idx[1]
    baselineConfidence = swarm.calculateBaselineConfidence(model, targetLabel, input)

    swarm.setSwarmAttributes(solu_plc, C1, C2, lowerBoundary, upperBoundary, blockSize)
    initialFitness = np.ones(numofMethods * numOfParticles, dtype='float')
    swarm.setInitialFitness(initialFitness)
    if verbosity >= 1:
        print('Model Prediction Before PSO= %s' % (pred))
        print('Baseline Confidence= %s' % (str(baselineConfidence)))

    swarm.initializeSwarmAndParticles(solu_plc, initialFitness, model, input, classes, j)
    return swarm, baselineConfidence, pred, targetLabel, dirPath, solu_plc


def mmpsoattack(input, N, dim, label, path, i, j):
    pred = object_function(model, input)
    input, lowerBoundary, upperBoundary = getBoundaries(input)
    print("Searching Advresarial Example for test sample %s..." % (i))

    swarm, baselineConfidence, pred, targetLabel, dirPath, solu_plc = Initialization(pred, input, model, i,
                                                                                     lowerBoundary, upperBoundary, j,
                                                                                     classes, path, label)

    if baselineConfidence == '':
        AfterAtkImage = re2Image(solu_plc, dataset, input, watermark, sl)
        AfterAtkImage.save(os.path.join(dirPath, 'AfterPSO_patch.png'))
        return True
    _, _, iterations, numberOfQueries, atkflag = swarm.searchOptimum(model, input, classes, j)
    finalFitness = swarm.bestFitness
    if atkflag:
        print("atk success:", label, " Now Label:", swarm.swarmLabel)
        AfterAtkImage = re2Image(swarm.swarmBestPosition, dataset, input, watermark, sl)
        AfterAtkImage.save(os.path.join(dirPath, 'AfterPSO.png'))
        swarm.cleanSwarm()
        return True
    result_flag = False
    swarm.cleanSwarm()
    return result_flag


if __name__ == '__main__':

    """
    采用新的搜索过程
    """
    count = 0
    count1 = 0
    dim = 3
    sl = 4

    data_dir = r'..\datasets\cluster_test_100'
    classes = sorted(os.listdir(data_dir))

    '''[resnet, alexnet, vgg, squeezenet, densenet, inception] '''
    model_name = 'resnet'
    model, input_size = getModel_X(model_name)
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor()
    ])
    traindata = ImageFolder(data_dir, transform=transform)

    watermark_logo = 'Cambridge'
    watermark, watermark_x1, watermark_y1, size = Open_logo(os.path.join(
        r'../logo/Cambridge.png'), sl, dataset)
    value_up_range = [200, input_size - watermark_x1, input_size - watermark_y1]
    value_down_range = [100, 0, 0]

    path = []
    filename = []

    AIM = False
    if AIM == True:
        aim_label = 'mountain'
        list_ = sorted(os.listdir(data_dir + '/' + aim_label))
        for mem in tqdm(list_):
            filename.append(mem)
            path.append(data_dir + '/' + aim_label + '/' + mem)
    else:
        for label in classes:
            list_ = sorted(os.listdir(data_dir + '/' + label))
            for mem in tqdm(list_):
                filename.append(mem)
                path.append(data_dir + '/' + label + '/' + mem)
    save_all_file_path(data_dir, ".JPEG")
    sampleList = sorted(random.sample(range(len(traindata)), 300))
    print('Attack list --> ', sampleList)

    for i in range(0, len(path)):
        for j in range(len(classes)):
            matchObj = re.search(classes[j], path[i])
            if matchObj:
                pred = getpred(path[i], classes[j], model, classes)
                if (pred == classes[j]):
                    start = time.time()
                    count = count + 1

                    input = Image.open(path[i])
                    if (mmpsoattack(input, numOfParticles, dim, classes[j], path[i], i, j)):
                        print(filename[i], "的攻击完成")
                        count1 += 1
                        print('共', count, "条数据, 成功率:", float(count1 / count), end='\n')
                    else:
                        print(filename[i], "未能攻击成功", '共', count, "条数据, 成功率:", float(count1 / count), end='\n')
                    end = time.time()
                    print('Time:{}ms'.format((end - start) * 1000))
                else:
                    print("Model has orierror on this sample, skip", i, end='\n')
