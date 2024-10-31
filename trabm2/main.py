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

# https://github.com/opencv/opencv_contrib/blob/4.x/samples/python2/seeds.py
def seeds(img_or, title, nsp, prir, levels, iter):

    img_in = cv2.cvtColor(img_or, cv2.COLOR_BGR2HSV)
    height, width, channels = img_in.shape

    num_superpixels = nsp # 6
    prior = prir # 4
    num_levels = levels # 30
    num_histogram_bins = 10
    seeds = cv2.ximgproc.createSuperpixelSEEDS(width, height, channels, num_superpixels, num_levels, prior, num_histogram_bins)
    color_img    = np.zeros((height,width,3), np.uint8)        
    color_img[:] = (0, 0, 255)

    seeds.iterate(img_in, iter)

    labels = seeds.getLabels()

    # labels output: use the last x bits to determine the color
    num_label_bits = 2
    labels &= (1<<num_label_bits)-1
    labels *= 1<<(16-num_label_bits)

    mask = seeds.getLabelContourMask(False)

    mask_inv = cv2.bitwise_not(mask)
    result_bg = cv2.bitwise_and(img_or, img_or, mask=mask_inv)
    result_fg = cv2.bitwise_and(color_img, color_img, mask=mask)
    result = cv2.add(result_bg, result_fg)

    cv2.imwrite(title, result)

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
    
    seeds(util.openimg('data/face1.jpg', color=True), 'results/seeds1.png', nsp=6, prir=4, levels=30, iter=2000)
    seeds(util.openimg('data/face2.jpg', color=True), 'results/seeds2.png', nsp=2, prir=3, levels=30, iter=2000)
    seeds(util.openimg('data/face2.jpg', color=True), 'results/hi_seeds2.png', nsp=200, prir=3, levels=30, iter=2000)
    seeds(util.openimg('data/face2.jpg', color=True), 'results/me_seeds2.png', nsp=20, prir=3, levels=30, iter=2000)
    seeds(util.openimg('data/face2.jpg', color=True), 'results/15_seeds2.png', nsp=15, prir=3, levels=30, iter=2000)
    seeds(util.openimg('data/face2.jpg', color=True), 'results/10_seeds2.png', nsp=10, prir=3, levels=30, iter=2000)
    seeds(util.openimg('data/face3.jpg', color=True), 'results/seeds3.png', nsp=3, prir=5, levels=30, iter=2000)
    seeds(util.openimg('data/face4.jpg', color=True), 'results/seeds4.png', nsp=5, prir=2, levels=30, iter=2000)
    seeds(util.openimg('data/face4.jpg', color=True), 'results/hi_seeds4.png', nsp=2000, prir=5, levels=30, iter=2000)