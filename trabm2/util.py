# SPDX-License-Identifier: 0BSD
import cv2   as cv2
import numpy as np
from skimage.util import random_noise

def salt_n_pepper(img_in):
    return np.array(random_noise(img_in, 's&p')*255, dtype=np.uint8)

def gaussian_noise(img_in):
    return np.array(random_noise(img_in, 'gaussian')*255, dtype=np.uint8)

def openimg(path, color=False):
    if color:
        return cv2.imread(path)
    else:
        return cv2.imread(path, 0)

def showimg(img_in, title=''):
    showimg.counter += 1

    if title == '':
        title = 'image' + str(showimg.counter)

    cv2.imshow(title, img_in)
    showimg.images.append(title)

showimg.counter = 0
showimg.images  = []

def pause():
    paused = True
    while paused:
        k = cv2.waitKey(100)
        if k == 27:
            cv2.destroyAllWindows()
            break        
        else:
#            for i in range(1, showimg.counter+1):
            for n in showimg.images:
                if cv2.getWindowProperty(n, cv2.WND_PROP_VISIBLE) < 1:
                    paused = False
                    showimg.images.clear()
                    break  
    cv2.destroyAllWindows()

def showimg_n_block(img_in):
    cv2.imshow('image', img_in)
    while True:
        k = cv2.waitKey(100)
        if k == 27:
            cv2.destroyAllWindows()
            break        
        elif cv2.getWindowProperty('image',cv2.WND_PROP_VISIBLE) < 1:        
            break        
    cv2.destroyAllWindows()