# SPDX-License-Identifier: 0BSD

import numpy as np
import cv2  as cv2
import trabm2.util as util
from tqdm  import tqdm
from queue import Queue

# SkinSegmentation do Professor
# https://medium.com/swlh/human-skin-color-classification-using-the-threshold-classifier-rgb-ycbcr-hsv-python-code-d34d51febdf8
# Link: https://github.com/VielF/ColabProjects/blob/main/PDI/SkinSegmentation.py

def RGB_Threshold(bgr):
    b = float(bgr[0])
    g = float(bgr[1])
    r = float(bgr[2])

    E1 = r > 95 and g > 40 and b > 20 and (max(r, g, b) - min(r, g, b)) > 15 and abs(r-g) > 15 and r > g and r > b
    E2 = r > 220 and g > 210 and b > 170 and abs(r - g) <= 15 and b < r and b < g

    return E1 or E2

def YCrCb_Threshold(yCrCb):
    y = float(yCrCb[0])
    Cr = float(yCrCb[1])
    Cb = float(yCrCb[2])

    E1 = Cr <=  1.5862 * Cb + 20
    E2 = Cr >=  0.3448 * Cb + 76.2069
    E3 = Cr >= -1.5652 * Cb + 234.5652 # -4.5652
    E4 = Cr <= -1.1500 * Cb + 301.75
    E5 = Cr <= -2.2857 * Cb + 432.85

    return E1 and E2 and E3 and E4 and E5

def HSV_Threshold(hsv, arg):
    if arg is None:
        return 0 <= hsv[0] and hsv[0] <= 50 and 255*0.23 <= hsv[1] and hsv[1] <= 255*0.68 # hsv[0] < 50 and hsv[0] > 150
    else:
        hlower = arg[0]
        hupper = arg[1]
        return hlower <= hsv[0] and hsv[0] <= hupper and 255*0.23 <= hsv[1] and hsv[1] <= 255*0.68

def Threshold(bgra, hsv, yCrCb, arg):
    return RGB_Threshold(bgra) and HSV_Threshold(hsv, arg) and YCrCb_Threshold(yCrCb)

# https://arxiv.org/ftp/arxiv/papers/1708/1708.02694.pdf
def SkinSegmentation(img, mode='all', arg=None):

    result = np.copy(img)
    bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
    hsv = cv2.normalize(hsv, None, 0.0, 255.0, cv2.NORM_MINMAX, cv2.CV_32FC3)
    yCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    match mode:
        case 'all':
            for i in tqdm(range(img.shape[0])):
                for j in range(img.shape[1]):
                    if (not Threshold(bgra[i, j], hsv[i, j], yCrCb[i, j], arg)):
                        result[i, j, 0] = 0
                        result[i, j, 1] = 0
                        result[i, j, 2] = 0
        case 'rgb':
            for i in tqdm(range(img.shape[0])):
                for j in range(img.shape[1]):
                    if (not RGB_Threshold(bgra[i, j])):
                        result[i, j, 0] = 0
                        result[i, j, 1] = 0
                        result[i, j, 2] = 0
        case 'ycrcb':
            for i in tqdm(range(img.shape[0])):
                for j in range(img.shape[1]):
                    if (not YCrCb_Threshold(yCrCb[i][j])):
                        result[i, j, 0] = 0
                        result[i, j, 1] = 0
                        result[i, j, 2] = 0
        case 'hsv':
            for i in tqdm(range(img.shape[0])):
                for j in range(img.shape[1]):
                    if (not HSV_Threshold(hsv[i][j], arg)):
                        result[i, j, 0] = 0
                        result[i, j, 1] = 0
                        result[i, j, 2] = 0

    return result