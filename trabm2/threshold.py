# SPDX-License-Identifier: 0BSD
import numpy as np
import cv2   as cv2

def erosion(img_in, struct):
  img_out = np.zeros([img_in.shape[0], img_in.shape[1]], dtype=np.uint8)

  w = img_in.shape[0]
  h = img_in.shape[1]
  s = struct.shape[0] // 2

  mc = np.count_nonzero(struct)

  for y in range(h):
    for x in range(w):
      count = 0
      for t in range(-s, 1+s):
        for r in range(-s, 1+s):
          u = y + t
          v = x + r
          if (v <  0 or u <  0):
            continue
          if (v >= w or u >= h):
            continue
          if struct[r+s][t+s] == 255:
            if img_in[v][u] == 255:
              count += 1
      if count == mc:
        img_out[x][y] = 255

  return img_out

def dilatation(img_in, struct):
    ac  = (img_in + 1) * 255
    abc = erosion(ac, struct)
    ab  = (abc    + 1) * 255
    return ab

def binary_threshold(img_in, thresh):

    img_in = img_in.copy()

    for i in range(img_in.shape[0]):
        for j in range(img_in.shape[1]):

            if (img_in[i, j] > thresh):
                img_in[i, j] = 255
            else:
                img_in[i, j] = 0

    return img_in

def otsu_threshold(img_in):
    retval, img_otsu = cv2.threshold(np.uint8(img_in), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img_otsu