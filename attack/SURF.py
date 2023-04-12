import cv2
import matplotlib.pyplot as plt
import random
import numpy as np
import numpy.random
from torchvision import transforms
from PIL import Image


def main(img, N):

    xp = np.zeros(N, np.float32)
    yp = np.zeros(N, np.float32)
    keypoint = []
    candicate = numpy.random.choice(N * 2, size=N, replace=False)
    for i in range(1, N):
        hes = 200 * i
        surf = cv2.xfeatures2d.SURF_create(hes)
        keypoint, _ = surf.detectAndCompute(img, None)
        if len(keypoint) < 150:
            break
    for i in range(len(candicate)):
        try:
            xp[i] = keypoint[candicate[i]].pt[0]
            yp[i] = keypoint[candicate[i]].pt[1]
        except Exception as e:
            return xp, yp
    return xp, yp


if __name__ == '__main__':
    imagepath = r'E:\PaperLearning\PyProject\Adv-watermark\demos\attacks20220304' \
                r'\Results\Reviewer#3\2772\n03201208_25225.JPEG'
    N = 50
    dim = 3
    X1 = []
    img = cv2.imread(imagepath)
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
    ])
    img = transform(Image.fromarray(img))
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
