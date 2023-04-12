import dlib
import numpy as np
import random
from torchvision import transforms


def main(img, N):
    xp = np.zeros(N, np.float32)
    yp = np.zeros(N, np.float32)
    predictor_path = "./shape_predictor_81_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    sampleList = sorted(random.sample(range(70), N))
    predicator = dlib.shape_predictor(predictor_path)
    dets = detector(img, 1)
    i = 0
    for k, d in enumerate(dets):
        # 坐标获取
        lanmarks = np.array([np.array([p.x, p.y]) for p in predicator(img, d).parts()]).tolist()
        for p in lanmarks:
            if (lanmarks.index(p) in sampleList) & (i < N):
                xp[i] = p[0]
                yp[i] = p[1]
                i += 1
    return xp, yp
