# SPDX-License-Identifier: 0BSD

import numpy as np
import cv2  as cv2
import trabm2.util as util
from tqdm  import tqdm
from queue import Queue

# SkinSegmentation do Professor
# Link: https://github.com/VielF/ColabProjects/blob/main/PDI/SkinSegmentation.py

def RGB_Threshold(bgr):
    b = float(bgr[0])
    g = float(bgr[1])
    r = float(bgr[2])

#   E1 = ? # add eqquaintion in paper
#   E2 = ? # add eqquaintion in paper

    return E1 or E2

def YCrCb_Threshold(yCrCb):
    y = float(yCrCb[0])
    Cr = float(yCrCb[1])
    Cb = float(yCrCb[2])

#   E1 = ? # add eqquaintion in paper
#   E2 = ? # add eqquaintion in paper
#   E3 = ? # add eqquaintion in paper
#   E4 = ? # add eqquaintion in paper
#   E5 = ? # add eqquaintion in paper

    return E1 and E2 and E3 and E4 and E5

#def HSV_Threshold(hsv):
#   return ? ? # add eqquaintion in paper

def Threshold(bgra, hsv, yCrCb):
    b  = float(bgra[0])
    g  = float(bgra[1])
    r  = float(bgra[2])
    a  = float(bgra[3])
    y  = float(yCrCb[0])
    Cr = float(yCrCb[1])
    Cb = float(yCrCb[2])
    h  = float(hsv[0])
    s  = float(hsv[1])
    v  = float(hsv[2])

    E1 = 0.0 <= h and h <= 50.0 and 0.23 <= s and s <= 0.68 and r > 95 and g > 40 and b > 20 and r > g and r > b and abs(r - g) > 15 and a > 15
    E2 = r > 95 and g > 40 and b > 20 and r > g and r > b and abs(r - g) > 15 and a > 15
    E2 = E2 and Cr > 135
    E2 = E2 and Cb > 85 and y > 80 and Cr <= (1.5862*Cb)+20
    E2 = E2 and Cr >= (0.3448*Cb)+76.2069
    E2 = E2 and Cr <= (-1.15*Cb)+301.75
    E2 = E2 and Cr <= (-2.2857*Cb)+432.85
    return E1 or E2

# https://arxiv.org/ftp/arxiv/papers/1708/1708.02694.pdf
def SkinSegmentation(img):
    result = np.copy(img)
    bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
    hsv = cv2.normalize(hsv, None, 0.0, 255.0, cv2.NORM_MINMAX, cv2.CV_32FC3)
    yCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    for i in tqdm(range(img.shape[0])):
        for j in range(img.shape[1]):
            if (not Threshold(bgra[i, j], hsv[i, j], yCrCb[i, j])):
                result[i, j, 0] = 0
                result[i, j, 1] = 0
                result[i, j, 2] = 0

#   cv2.imwrite(imgNameOut, result)
    return result