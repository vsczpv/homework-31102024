# SPDX-License-Identifier: 0BSD
import trabm2.util      as util
import trabm2.threshold as thresh
import trabm2.skinseg   as skinseg
import cv2 as cv2
import time

def threshes(img_orig, img_dsalt, img_dgaus):
    
    img_ootsu = thresh.otsu_threshold(img_orig)
    img_sotsu = thresh.otsu_threshold(img_dsalt)
    img_gotsu = thresh.otsu_threshold(img_dgaus)

    util.showimg(img_ootsu, title='OOtsu')
    for i in [112, 120, 128, 144]:
        util.showimg(thresh.binary_threshold(img_orig, i), title='OBasica' + str(i))
    util.pause()

    util.showimg(img_sotsu, title='SOtsu')
    for i in [112, 120, 128, 144]:
        util.showimg(thresh.binary_threshold(img_dsalt, i), title='SBasica' + str(i))
    util.pause()

    util.showimg(img_gotsu, title='GOtsu')
    for i in [112, 120, 128, 144]:
        util.showimg(thresh.binary_threshold(img_dgaus, i), title='GBasica' + str(i))
    util.pause()

    util.showimg(img_sotsu, title='SOtsu')
    util.showimg(img_gotsu, title='GOtsu')
    util.showimg(thresh.binary_threshold(img_dsalt, 120), title='SBasica120')
    util.showimg(thresh.binary_threshold(img_dgaus, 120), title='GBasica120')
    util.pause()

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