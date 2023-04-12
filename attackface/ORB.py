import cv2
import matplotlib.pyplot as plt
import random
import numpy as np
import numpy.random
from torchvision import transforms


def main(img, N, inputsize):
    # img = img.resize(inputsize)
    xp = np.zeros(N, np.float32)
    yp = np.zeros(N, np.float32)
    for i in range(1, N):
        num = 50 * i
        orb = cv2.ORB_create(nfeatures=num)
        keypoint, des = orb.detectAndCompute(img, None)
        if len(keypoint) > 100:
            break
    candicate = numpy.random.choice(N * 2, size=N, replace=False)
    for i in range(len(candicate)):
        try:
            xp[i] = keypoint[candicate[i]].pt[0]
            yp[i] = keypoint[candicate[i]].pt[1]
        except Exception as e:
            return xp, yp
    return xp, yp


if __name__ == '__main__':
    imagepath = 'E:\\PaperLearning\\datasets\\imagenet_mini\\train\\n01614925\\n01614925_9309.JPEG'
    # imagepath = 'C:\\Codes\\datasets\\imagenet_mini\\train\\n01614925\\n01614925_11265.JPEG'
    N = 50
    dim = 3
    X1 = []
    img = cv2.imread(imagepath)
    img = cv2.resize(img, (224, 224))
    xp, yp = main(img, N)
    print(xp, yp)


    def init_position_select_range(x):
        if x > 224 - 56:
            return 224 - 56
        else:
            return x


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
    print(np.array(X1))
