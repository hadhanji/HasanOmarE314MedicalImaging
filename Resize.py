# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 18:29:45 2021

@author: Hasan
"""
from os import walk


import cv2
import glob
import os


f = []
for (dirpath, dirnames, filenames) in walk('C:/ECEN314FinalProject/Covid19-dataset/test/Viral Pneumonia/'):
    f.extend(filenames)
    break 
print(f)

for img in f:
    image = cv2.imread('C:/ECEN314FinalProject/Covid19-dataset/test/Viral Pneumonia/'+img)
    imgResized = cv2.resize(image,(50,50))
    cv2.imwrite('C:/FinalProjectResized/Covid19-dataset/test/Viral Pneumonia/' + img, imgResized)
    cv2.imshow('image', imgResized)


