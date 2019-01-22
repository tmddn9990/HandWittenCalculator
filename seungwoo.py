import cv2
import os
import numpy as np
import PIL
import matplotlib.pyplot as plt
import scipy.misc as sp
from  sklearn import preprocessing

path='C:/Users/tmddn/source/repos/tensorflow/tensorflow/image'


os.chdir(path)
features=[]

for fileName in os.listdir():
    img=cv2.imread(fileName,cv2.IMREAD_GRAYSCALE)
    print(fileName)
    features.append(img)
    

image_set={}
image_set["features"]=np.asarray(features)


np.save("imageset",image_set)