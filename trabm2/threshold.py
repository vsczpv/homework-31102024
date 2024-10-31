# SPDX-License-Identifier: 0BSD
import numpy as np
import cv2   as cv2

def erosion(imagem, kernel=[[1,1,1],[1,1,1],[1,1,1],[1,1,1]]):
    imagem_erodida = np.zeros(imagem.shape, dtype=np.uint8)
    kernel = np.array(kernel)
    offI = int(np.floor(kernel.shape[1]/2.0))# Centro vertical do kernel
    imparI = kernel.shape[1]%2 # Se for impar, é somado 1 pelo motivo comentado dentro dos lassos for
    offJ = int(np.floor(kernel.shape[0]/2.0))# Centro horizontal do kernel
    imparJ = kernel.shape[0]%2
    for i in range(offI, int(imagem.shape[1]) - offI):# Executa apenas aonde o Kernel não sai da imagem
        for j in range(offJ, int(imagem.shape[0]) - offJ):
            if (kernel * imagem[int(j - offJ):int(j + offJ + imparJ), int(i - offI):int(i + offI + imparI)]).all() > 0:# Se nenhum for zero, executa
                                              #Se fizer (i - offI):(i+ offI), ele retorna i-offI e I, o ponto de parada não entra no cálculo
                imagem_erodida[j, i] = 255
            else:
                imagem_erodida[j, i] = 0

    return imagem_erodida

def dilatation(img_in, struct):
    ac  = (img_in + 1) * 255
    abc = erosion(ac, struct)
    ab  = (abc    + 1) * 255
    return ab

def opening(img_in, struct):
    C = erosion(img_in, struct)
    return dilatation(C, struct)

def closure(img_in, struct):
    C = dilatation(img_in, struct)
    return erosion(C, struct)

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