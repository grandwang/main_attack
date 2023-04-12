# -*- coding: utf-8 -*-
import os

import numpy as np
import torch
import random
from copy import deepcopy
from itertools import product
from Utilities import *
from particle import particle
from numpy.random import rand
import gc
from Dataset import returnDimensions
from data_process import cal_sigema, sigfun, sigfun_quo


class Swarm:
    bestParticleID = -1
    C1 = 0.0
    C2 = 0.0
    initialFitness = 0.0
    dLength = None
    numOfQueries = 0
    bestFitness = 0
    granularityCurrentLevel = 0

    def __init__(self, numOfParticles, model, ind, maxChange, dataset, dLength, verbosity, topN, targeted, queries,
                 numofMethods,
                 image, watermark, sl, input_size):
        self.numberOfParticles = numOfParticles
        self.targetModel = model
        self.classIND = ind
        self.dLength = dLength
        self.maxChange = maxChange
        self.dataset = dataset
        self.verbosity = verbosity
        self.topN = topN
        self.targeted = targeted
        self.queries = queries
        self.input_size = input_size

        self.numofMethods = numofMethods

        self.img = image
        self.watermark = watermark
        self.sl = sl
        self.swarmBestPosition = np.array([100, 100, 100])

        self.moveflag = False
        self.input_size = 224

    def cleanSwarm(self):
        del self.numberOfParticles
        del self.bestFitness
        del self.bestPosition
        del self.currentPosition
        del self.inputX
        del self.Particles

    def setCurrentPosition(self, newPosition):
        self.currentPosition = deepcopy(newPosition)

    def setBestPosition(self, newPosition):
        self.bestPosition = deepcopy(newPosition)

    def setInitialFitness(self, initialFitness):
        self.initialFitness = initialFitness

    def setCs(self, C1, C2):
        self.C1 = C1
        self.C2 = C2

    def getImageDimensions(self):
        self.height, self.width, self.channels = returnDimensions(self.dataset)

    def setSwarmAttributes(self, x, C1, C2, lowerBoundary, upperBoundary, blockSize):
        self.setCs(C1, C2)
        self.setBestPosition(x)
        self.setCurrentPosition(x)
        self.setInputX(x)
        self.setBoundaries(lowerBoundary, upperBoundary)

        self.maxChangeLower = np.subtract(self.inputX, self.maxChange)
        self.maxChangeUpper = np.add(self.inputX, self.maxChange)
        self.blockSize = blockSize

        self.sigema_max = 1
        self.sigema_min = 0.1
        self.P = np.zeros(len(self.currentPosition))
        self.P_quo = np.zeros(len(self.currentPosition))
        self.fitP = float('inf') * np.ones(self.numofMethods * self.numberOfParticles, dtype='float')
        self.initfitP = float('inf') * np.ones(self.numofMethods * self.numberOfParticles, dtype='float')
        self.fitG = float('inf') * np.ones(1, dtype='float')
        self.initfitG = float('inf') * np.ones(1, dtype='float')

        self.oriclass_rate = []
        self.atkclass_rate = []

    def setBoundaries(self, lowerBoundary, upperBoundary):
        self.lowerBoundary = lowerBoundary
        self.upperBoundary = upperBoundary

    def set_init_velocity(self):
        v = np.zeros(len(self.currentPosition), dtype='float')
        v_max = (self.maxChange - 0) / 2
        v_min = - v_max
        for k in range(len(self.currentPosition)):
            v[k] = v_min + random.uniform(v_min, v_max)
        return v

    def setBestFitnessScore(self, newScore):
        self.bestFitness = newScore

    def reshapeAndCompare(self, positions):
        z = re2Watermark(positions, self.dataset, self.img, self.watermark, self.sl, self.numofMethods,
                         self.numberOfParticles, self.input_size)
        diff = compareImages(self.inputX, positions, len(positions))
        return z, diff

    def returnTopNPred(self, pred):
        if self.topN > 1:
            pred = np.argsort(pred)[:][::-1][:self.topN]
        elif self.topN == 1:
            pred = np.argmax(pred)
        return pred

    def setInputX(self, newPosition):
        self.inputX = deepcopy(newPosition)

    def calculateBaselineConfidence(self, net, ind, input):
        proBaseline = predictSample(net, input)[0]
        self.bestProba = proBaseline
        self.numOfQueries = self.numOfQueries + 1
        if type(ind) is np.int64 or type(ind) is int:
            proBaseline = proBaseline[ind]
        elif type(ind) is np.ndarray:
            avgProba = []
            for i, x in enumerate(ind):
                avgProba.append(proBaseline[x])
            proBaseline = avgProba
        self.baselineConfidence = proBaseline
        return proBaseline

    def initX(self, net, xs, label, input, classes, watermark, sl):
        alpha = 0.9
        beta = 1 - alpha
        img = input.convert('RGB')
        _, fstconfi = ec_main(net, img, '', classes)

        data_transform = transforms.Compose(
            [transforms.CenterCrop(self.input_size),
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
            predict = torch.softmax(output, dim=1)
            atkconfi, atkpre = torch.max(predict, dim=1)

        if any(atkpre != label):
            cost = int(0)
            return cost
        else:
            err = 1 - fstconfi
            cost = alpha * err + beta * (fstconfi - atkconfi)
            return cost

    def generateISSIndicesDictionary(self):
        self.issIndices = {}
        cd = list(product([0, -1, 1], repeat=self.channels))
        if self.channels == 3:
            cd = [(a, b, c) for a, b, c in cd if len(np.nonzero([a, b, c])[0]) == 1]
        elif self.channels == 1:
            cd = [(a) for a in cd if not a[0] == 0]
        for i in range(len(self.ISS)):
            self.issIndices[i] = deepcopy(cd)

    def generateIndividualSearchSpaces(self):
        self.granularityLevels = halfsOfN(self.blockSize, 2)
        self.ISS = splitSoluIntoNbyNRegions(self.numofMethods, self.numberOfParticles, 3,
                                            self.granularityLevels[self.granularityCurrentLevel])
        self.generateISSIndicesDictionary()

        self.updateChangeRate()
        if self.verbosity >= 2:
            print("Blocksize= %s" % (self.granularityLevels[self.granularityCurrentLevel]))
            print("Change Rate Per Particle= %s" % (self.changeRate))

    def increaseISSGranularity(self, net, input, classes, j):
        if self.granularityCurrentLevel + 1 < len(self.granularityLevels):
            self.granularityCurrentLevel = self.granularityCurrentLevel + 1

            self.ISS = splitSoluIntoNbyNRegions(self.numofMethods, self.numberOfParticles, 3,
                                                self.granularityLevels[self.granularityCurrentLevel])
            self.generateISSIndicesDictionary()
            self.Particles = self.initializeParticles(self.bestPosition, net, input, classes, j)
            self.Check()
            if self.verbosity >= 2:
                print("Blocksize= %s" % (self.granularityLevels[self.granularityCurrentLevel]))
                print("Change Rate Per Particle= %s" % (self.changeRate))
        else:
            if self.granularityCurrentLevel + 1 == len(self.granularityLevels):
                self.generateISSIndicesDictionary()
                self.resetParticles(net, input, classes, j)
                self.Check()
        return

    def updateChangeRate(self):
        lenOfIndices = len(self.issIndices)
        self.changeRate = lenOfIndices

    def initializeSwarm(self, sample):
        self.Particles = [particle] * self.numberOfParticles
        self.setBestPosition(sample)
        self.setCurrentPosition(sample)
        self.setBestFitnessScore(self.initialFitness)

    def initializeSwarmAndParticles(self, solu_plc, fitness, net, input, classes, j):
        self.pastFitness = []
        self.getImageDimensions()
        self.flag = False
        if self.targeted == True:
            self.swarmLabel = -1
        else:
            self.swarmLabel = self.classIND
        self.previousGranBest = deepcopy(self.inputX)
        self.initializeSwarm(solu_plc)
        self.generateIndividualSearchSpaces()
        self.Particles = self.initializeParticles(solu_plc, net, input, classes, j)
        self.Check()
        self.pastFitness.append(self.bestFitness)

    def initializeParticles(self, startingPosition, net, input, classes, j):
        particleList = []
        for particles in range(0, self.numberOfParticles):
            p = None
            p = particle(particles)
            p.setW()
            p.blocks = {}
            p.setInitBestFitnessScore(self.numofMethods, self.numberOfParticles)
            p.setCurrentPosition(startingPosition)
            p.setBestPosition(startingPosition)
            p.setNextPosition(startingPosition)
            p.currentVelocity = self.set_init_velocity()

            p, newProba, _ = self.randomizeParticle(startingPosition, p, net, input, classes, j)
            particleList.append(deepcopy(p))
            del p
        return particleList

    def fitScore(self, V, r1, r2, sigema_max, sigema_min, net, input, classes, j):
        sigema = cal_sigema(sigema_max, sigema_min, self.numOfQueries, self.queries)
        for i in range(len(V)):
            if r1[i] < sigfun(V[i], sigema):
                self.P[i] = self.currentPosition[i] + V[i]
            else:
                self.P[i] = self.currentPosition[i]
            if r2[i] > sigfun_quo(V[i], sigema):
                self.P_quo[i] = self.currentPosition[i] + V[i]
            else:
                self.P_quo[i] = self.currentPosition[i]
        P = self.P.reshape(self.numofMethods * self.numberOfParticles, 3)
        P_quo = self.P_quo.reshape(self.numofMethods * self.numberOfParticles, 3)

        self.fit, idx = fitnessScore(self, net, P, input, classes, self.watermark, self.sl,
                                     j)
        self.fit_quo, idx_quo = fitnessScore(self, net, P_quo, input, classes, self.watermark, self.sl, j)

        if self.fit == True:
            self.swarmLabel = idx
            return 0, idx
        elif self.fit_quo == True:
            self.swarmLabel = idx
            return 0, idx_quo
        else:
            self.bestPosition = self.bestPosition.reshape(self.numofMethods * self.numberOfParticles, 3)
            i = 0
            for elem, elem_quo in zip(self.fit, self.fit_quo):
                if elem < self.fitP[i]:
                    self.fitP[i] = elem
                    self.bestPosition[i] = P[i]
                if elem_quo < self.fitP[i]:
                    self.fitP[i] = elem_quo
                    self.bestPosition[i] = P_quo[i]
                if self.fitP[i] < self.fitG:
                    self.fitG = self.fitP[i]
                    self.swarmBestPosition = self.bestPosition[i]
                    if idx[i] != self.classIND:
                        self.swarmLabel = idx[i]
                i += 1
        self.currentPosition = self.bestPosition.flatten()
        self.bestPosition = np.full([self.numofMethods * self.numberOfParticles, 3], self.swarmBestPosition).flatten()
        return self.fitP, idx

    def moveParticleAndCalNewFitness(self, p, net, input, classes, j):

        v, r1, r2 = p.calculateV(self.bestPosition, self.numOfQueries, self.C1, self.C2, self.inputX,
                                 self.maxChangeLower, self.maxChangeUpper, self.queries, self.maxChange,
                                 lowerBound=self.lowerBoundary, upperBound=self.upperBoundary)
        newFitness, newPred = self.fitScore(v, r1, r2, self.sigema_max, self.sigema_min, net, input, classes, j)
        p.setBestFitnessScore(newFitness)

        p.setCurrentPosition(self.currentPosition)

        return newFitness, newPred

    def InitialFitScore(self, V, r1, r2, sigema_max, sigema_min, net, input, classes, j):

        newPred = []
        sigema = cal_sigema(sigema_max, sigema_min, self.numOfQueries, self.queries)
        for i in range(len(V)):
            if r1[i] < sigfun(V[i], sigema):
                self.P[i] = self.currentPosition[i]
            else:
                self.P[i] = self.currentPosition[i] + V[i]
            if r2[i] > sigfun_quo(V[i], sigema):
                self.P_quo[i] = self.currentPosition[i]
            else:
                self.P_quo[i] = self.currentPosition[i] + V[i]
        P = self.P.reshape(self.numofMethods * self.numberOfParticles, 3)
        P_quo = self.P_quo.reshape(self.numofMethods * self.numberOfParticles, 3)
        self.fit, idx = InitFitnessScore(self, net, P, input, classes, self.watermark, self.sl,
                                         j)
        self.fit_quo, idx_quo = InitFitnessScore(self, net, P_quo, input, classes, self.watermark, self.sl, j)
        i = 0
        for elem, elem_quo in zip(self.fit, self.fit_quo):
            newPred.append(idx[i])
            if elem < self.initfitP[i]:
                self.initfitP[i] = elem

            if elem_quo < self.initfitP[i]:
                self.initfitP[i] = elem_quo

            if self.initfitP[i] < self.initfitG:
                self.fitG = self.initfitP[i]

            i += 1

        return self.initfitP, numpy.array(newPred)

    def randomizeParticle(self, startingPosition, p, net, input, classes, j):
        iss = self.returnRandomSearchSpaces(p)
        if not iss:
            return p, p.currentFitness, 0
        indices = self.getRandomizationParameters(iss)
        temp = deepcopy(startingPosition)

        p.currentPosition = self.Randomize(indices, temp)

        r1_p = np.random.uniform(0.0, 1.0, len(p.currentPosition))
        r2_p = np.random.uniform(0.0, 1.0, len(p.currentPosition))
        newFitness, newProba = self.InitialFitScore(p.currentVelocity, r1_p, r2_p, self.sigema_max, self.sigema_min,
                                                    net, input, classes, j)

        p.setCurrentFitnessScore(newFitness)

        del temp
        return p, newFitness, newProba

    def randomizeParticle_reset(self, startingPosition, p, net, input, classes, j):
        iss = self.returnRandomSearchSpaces(p)
        if not iss:
            return p, p.currentFitness, 0
        indices = self.getRandomizationParameters(iss)
        temp = deepcopy(startingPosition)

        p.currentPosition = self.Randomize(indices, temp)

        r1_p = np.random.uniform(0.0, 1.0, len(p.currentPosition))
        r2_p = np.random.uniform(0.0, 1.0, len(p.currentPosition))
        newFitness, newProba = self.fitScore(p.currentVelocity, r1_p, r2_p, self.sigema_max, self.sigema_min, net,
                                             input, classes, j)

        p.setCurrentFitnessScore(newFitness)
        del temp
        return p, newFitness, newProba

    def Randomize(self, indices, temp):
        temp = np.clip(np.add(indices, temp), self.maxChangeLower, self.maxChangeUpper)
        return temp

    def returnRandomSearchSpaces(self, p):
        if len(p.blocks) == len(self.ISS):
            p.blocks = {}
        keys = random.sample(list(self.issIndices.keys()),
                             self.changeRate if self.changeRate <= len(self.issIndices) else len(
                                 self.issIndices))
        iss = self.iterateKeys(p, keys)
        if len(self.issIndices) == 0:
            self.flag = True
        if len(iss) == 0:
            return []
        else:
            return iss

    def iterateKeys(self, p, keys):
        iss = {}
        for key in keys:
            if len(iss) >= self.changeRate:
                break
            if len(self.issIndices[key]) == 0:
                self.issIndices.pop(key, None)
                continue
            if key in p.blocks:
                keysInParticle = deepcopy(p.blocks[key])
                if self.channels == 3:
                    keysInParticle = [tuple(np.multiply((a, b, c), d)) for a, b, c in keysInParticle for d in [1, -1]]
                elif self.channels == 1:
                    keysInParticle = [tuple(np.multiply(a, d)) for a in keysInParticle for d in [1, -1]]
                directionsToSample = list(set(self.issIndices[key]).difference(keysInParticle))
            else:
                p.blocks[key] = []
                directionsToSample = self.issIndices[key]
            if not directionsToSample:
                continue
            lowLevel = random.choice(directionsToSample)
            iss[key] = lowLevel
            p.blocks[key].append(lowLevel)
            self.issIndices[key].remove(lowLevel)
            if len(self.issIndices[key]) == 0:
                self.issIndices.pop(key, None)
        return iss

    def getRandomizationParameters(self, searchSpace):
        indices = [0] * self.dLength
        for iss in searchSpace:
            directions = searchSpace[iss]
            step = 1.0
            for i in self.ISS[iss]:
                indices[i] = directions[i % self.channels] * self.maxChange * step
        return indices

    def Move(self, net, input, classes, j):
        for p in self.Particles:
            newFitness, newProba = self.moveParticleAndCalNewFitness(p, net, input, classes, j)
            if np.size(newFitness) == 1:
                return True
        return False

    def resetParticles(self, net, input, classes, j):
        for p in self.Particles:
            _, newFitness, _ = self.randomizeParticle_reset(p.currentPosition, p, net, input, classes, j)
            if np.size(newFitness) == 1:
                self.moveflag = True

    def runSearch(self, net, input, classes, j):
        if self.moveflag == True:
            return self.moveflag
        self.moveflag = self.Move(net, input, classes, j)

        if self.labelCheck() == False or self.numOfQueries >= self.queries:
            return False
        self.pastFitness.append(self.bestFitness)

    def Check(self):
        self.swarmLabel = self.returnTopNPred(self.bestProba)
        if np.size(self.swarmLabel) != 1:
            return

    def searchOptimum(self, net, input, classes, j):
        iteration = []
        if self.labelCheck():
            print("直接命中 ", " numOfQueries:", self.numOfQueries, " Ori Class: ", j)
            return self.bestPosition, self.bestFitness, 1, self.numOfQueries, True
        while self.queries > self.numOfQueries:
            atkflag = self.runSearch(net, input, classes, j)
            self.numOfQueries = self.numOfQueries + 1
            # self.mid_save(j)
            if atkflag == True:
                print("numOfQueries:", self.numOfQueries, " Ori Class, ", j)
                return self.bestPosition, self.bestFitness, iteration, self.numOfQueries, atkflag
            iteration.append(self.numOfQueries)
        print("numOfQueries:", self.numOfQueries, " Ori Class, ", j)
        return self.bestPosition, self.bestFitness, iteration, self.numOfQueries, ""

    def mid_save(self, j):
        midImage = re2Image(self.bestPosition, 'imagenet224', self.img, self.watermark, 4)
        savepath = os.path.join('C:/Codes/main_attack/reviewer#3/comment#6/',
                                str(j) + '_' + str(self.numOfQueries) + '_AfterMISPSO.png')
        midImage.save(savepath, quality=100)
        del midImage

    def labelCheck(self):
        if self.targeted == True:
            if not 'imagenet' in self.dataset and self.swarmLabel == self.classIND:
                return True
            elif 'imagenet' in self.dataset:
                if self.topN == 1:
                    if self.swarmLabel == self.classIND:
                        return True
                    else:
                        return False
                elif self.topN > 1:
                    if not any(elem in self.classIND[1:] for elem in self.swarmLabel[1:]) and self.swarmLabel[0] == \
                            self.classIND[0]:
                        return True
                    else:
                        return False
            else:
                return False

        elif self.targeted == False:  # 非指向
            if self.topN == 1:
                if np.size(self.swarmLabel) != 1:
                    return
                elif self.swarmLabel == self.classIND:
                    return False
                elif not self.swarmLabel == self.classIND:
                    return True
            elif self.topN > 1:
                if any(elem in self.classIND for elem in self.swarmLabel):
                    return False
                else:
                    return True
