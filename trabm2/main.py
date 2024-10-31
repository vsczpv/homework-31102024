# SPDX-License-Identifier: 0BSD
import trabm2.util      as util
import trabm2.threshold as thresh
import trabm2.skinseg   as skinseg
import trabm2.kmeans    as kmeans
from tqdm  import tqdm
import cv2 as cv2
import time
import numpy as np

def treat_fingerprint(img_in, struct=np.array([[1,1,1],[1,1,1],[1,1,1]])):
    img_in = (img_in + 1) * 255
    res = thresh.closure(thresh.opening(img_in, struct), struct)
    return (res + 1) * 255

def threshes(img_orig, img_dsalt, img_dgaus):
    
    img_ootsu = thresh.otsu_threshold(img_orig)
    img_sotsu = thresh.otsu_threshold(img_dsalt)
    img_gotsu = thresh.otsu_threshold(img_dgaus)

    cv2.imwrite('results/dig_ootsu.png', img_ootsu)
    cv2.imwrite('results/dig_treat_ootsu.png', treat_fingerprint(img_ootsu))
    for i in tqdm([112, 120, 128, 144]):
        nrm = thresh.binary_threshold(img_orig, i)
        cv2.imwrite('results/dig_obasic_' + str(i) + '.png', nrm)
        alt = treat_fingerprint(nrm)
        cv2.imwrite('results/dig_treat_obasic_' + str(i) + '.png', alt)

    cv2.imwrite('results/dig_sotsu.png', img_sotsu)
    cv2.imwrite('results/dig_treat_sotsu.png', treat_fingerprint(img_sotsu))
    for i in tqdm([112, 120, 128, 144]):
        nrm = thresh.binary_threshold(img_dsalt, i)
        cv2.imwrite('results/dig_sbasic_' + str(i) + '.png', nrm)
        alt = treat_fingerprint(nrm)
        cv2.imwrite('results/dig_treat_sbasic_' + str(i) + '.png', alt)

    cv2.imwrite('results/dig_gotsu.png', img_gotsu)
    cv2.imwrite('results/dig_treat_gotsu.png', treat_fingerprint(img_gotsu))
    for i in tqdm([112, 120, 128, 144]):
        nrm = thresh.binary_threshold(img_dgaus, i)
        cv2.imwrite('results/dig_gbasic_' + str(i) + '.png', nrm)
        alt = treat_fingerprint(nrm)
        cv2.imwrite('results/dig_treat_gbasic_' + str(i) + '.png', alt)


def main():
    img_orig  = util.openimg('data/digital.jpg')
    img_dsalt = util.openimg('data/digital_salt.png')  # util.salt_n_pepper (img_orig)
    img_dgaus = util.openimg('data/digital_gauss.png') # util.gaussian_noise(img_orig)

    threshes(img_orig, img_dsalt, img_dgaus)

    for t in ['all', 'rgb', 'ycrcb', 'hsv']:
        for i in [1, 2, 3, 4]:
            img = skinseg.SkinSegmentation(util.openimg('data/face' + str(i) + '.jpg', color=True), mode=t)
            cv2.imwrite('results/skinseg_' + t + '_' + str(i) + '.png', img)

    for t in ['all', 'hsv']:
        for i in [1, 2, 3, 4]:
            img = skinseg.SkinSegmentation(util.openimg('data/face' + str(i) + '.jpg', color=True), mode=t, arg=[0,20])
            cv2.imwrite('results/alt_skinseg_' + t + '_' + str(i) + '.png', img)

    for i in [1, 2, 4]:
        path = 'data/face' + str(i) + '.jpg'
        for j in [2, 3, 4, 5, 6]:
            kmeans.KMeans3D(util.openimg(path, color=True), k=j, max_iterations=30,  title=str(i))
            kmeans.KMeans3D(util.openimg(path, color=True), k=j, max_iterations=100, title=str(i))
    