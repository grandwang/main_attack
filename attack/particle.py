import numpy as np
from copy import deepcopy
from math import cos, pi
import gc

# from Utilities import reduceSearchSpace

class particle:
    def __init__(self, particleid=0):
        self.particleID = particleid
        self.bestFitness = []
        self.pastFitness = []
        self.currentPosition = None
        self.nextPosition = None
        self.bestPosition = None
        self.currentVelocity = None
        self.currentFitness = None#0
        self.nextVelocity = None
        self.blocks = {}

    def setNextPosition(self, newPosition):
        self.nextPosition = deepcopy(newPosition)

    def setCurrentPosition(self, newPosition):
        self.currentPosition = deepcopy(newPosition)

    def setBestPosition(self, newPosition):
        self.bestPosition = deepcopy(newPosition)

    def setBestFitnessScore(self, newScore):
        self.bestFitness = newScore

    def setInitBestFitnessScore(self, numofMethods, numofParticle):
        self.bestFitness = np.ones(numofMethods * numofParticle, dtype='float')

    def setCurrentFitnessScore(self, newScore):
        self.currentFitness = newScore

    def setW(self):
        self.wEND = 0.0
        self.wSTART = 1.0

    def cleanParticle(self):
        del self.particleID
        del self.currentPosition
        del self.nextPosition
        del self.bestPosition
        del self.bestFitness
        del self.currentVelocity
        del self.currentFitness
        gc.collect()

    def push(self, fitness, v):
        self.pastFitness.append((fitness, v))
        #按照fit score 排序
        self.pastFitness.sort(key=lambda x: x[0], reverse=True)

    def popHigh(self):
        indices = deepcopy(self.pastFitness[0][1])
        fitness = self.pastFitness[0][0]
        self.pastFitness = self.pastFitness[1:]
        return fitness, indices

    def peekHigh(self):
        if self.pastFitness:
            return self.pastFitness[0][0]
        else:
            return 0

    def popLow(self):
        indices = deepcopy(self.pastFitness[-1][1])
        fitness = self.pastFitness[-1][0]
        self.pastFitness = self.pastFitness[:-1]
        return fitness, indices

    def peekLow(self):
        if self.pastFitness:
            return self.pastFitness[-1][0]
        else:
            return 0

    def printParticleInformation(self):
        print('Particle %s -- Best Fitness %s \n' % (str(self.particleID), str(self.bestFitness)))

    def Velocity(self, swarmBestPosition, numOfQueries, C1, C2, maxQueries, maxChange):
        self.W = self.calculateW(numOfQueries, swarmBestPosition, self.wSTART, self.wEND, maxQueries)
        # part1 个体
        self.r1 = np.random.uniform(0.0, 1.0, len(self.bestPosition))
        particleBestDelta = np.multiply(np.multiply(np.subtract(self.bestPosition, self.currentPosition),
                                                    self.r1), C1)
        # part2 集群
        self.r2 = np.random.uniform(0.0, 1.0, len(swarmBestPosition))
        swarmBestDelta = np.multiply(np.multiply(np.subtract(swarmBestPosition, self.currentPosition),
                                                 self.r2), C2)
        #pb+gb
        deltas = np.add(np.asarray(particleBestDelta), np.asarray(swarmBestDelta))
        # part0 计算w*vt
        v =np.clip(np.add(self.W * np.clip(self.currentVelocity, -1 * maxChange, maxChange), deltas),-1 * maxChange, maxChange)
        self.nextVelocity = deepcopy(v)
        self.currentVelocity = deepcopy(v)
        return v

    def calculateV(self, swarmBestPosition, numOfQueries, C1, C2, inputX, maxChangeLower, maxChangeUpper,
                              maxQueries, maxChange, lowerBound, upperBound):
        # 更新V
        v = self.Velocity(swarmBestPosition, numOfQueries, C1, C2, maxQueries, maxChange)
        # 更新x
        # self.updateNextPostion(v, maxChangeLower, maxChangeUpper)
        # 更新粒子的CurrentPosition
        # self.setCurrentPosition(self.nextPosition)
        return v, self.r1, self.r2

    def calculateK(self, q, maxQueries):
        constrictionFactor = ((cos((pi / maxQueries) * q)) + 2.5) / 4
        return constrictionFactor

    def calculateW(self, numOfQueries, swarmBestPosition, w_max, w_min, maxQueries):
        #根据迭代次数更新W
        W = w_max - (w_max - w_min) * (numOfQueries / maxQueries)
        return W

    def updateNextPostion(self, v, maxChangeLower, maxChangeUpper):
        #下一个坐标位置 lowerBound&upperBound 下界与上界
        self.nextPosition = np.clip(np.add(v, self.currentPosition), maxChangeLower, maxChangeUpper)



    def getPerturbParameters(self, patches, inputX):
        #水印攻击不需要添加扰动，这个方法改为添加水印后图片转为ndarray的方法
        indices = [0] * len(inputX)
        for iss, vel in patches:
            for i in iss:
                indices[i] = vel
        return indices
