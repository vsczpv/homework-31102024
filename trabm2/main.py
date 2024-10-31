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
    img_dsalt = util.salt_n_pepper (img_orig)
    img_dgaus = util.gaussian_noise(img_orig)

#   util.showimg(img_orig,  title='Original')
#   util.showimg(img_dgaus, title='GaussianNoise')
#   util.showimg(img_dsalt, title='SaltnPepper')
#   util.pause()

#   threshes(img_orig, img_dsalt, img_dgaus)    
#   util.pause()

    for i in range(1, 5):
        util.showimg(skinseg.SkinSegmentation(util.openimg('data/face' + str(i) + '.jpg', color=True)), title='SkinSeg' + str(i))

    util.pause()