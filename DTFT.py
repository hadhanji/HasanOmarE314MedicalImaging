# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 20:15:41 2021

@author: Hasan
"""
from os import walk

import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
import glob
import os

f = []
for (dirpath, dirnames, filenames) in walk('C:/FinalProjectResized/Covid19-dataset/test/Normal/'):
    f.extend(filenames)
    break 
print(f)

for image in f:
    img = cv.imread('C:/FinalProjectResized/Covid19-dataset/test/Normal/' + image,0)
    dft = cv.dft(np.float32(img),flags = cv.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    magnitude_spectrum = 20*np.log(cv.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
    
    #plt.subplot(121),plt.imshow(img, cmap = 'gray')
    #plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.axis('off')
    #plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.savefig("C:/ECEN314FinalProjectFourier/Covid19-dataset/test/Normal/" + image, bbox_inches = 'tight')
    plt.show()
