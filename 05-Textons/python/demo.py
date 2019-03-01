#!/usr/bin/ipython3

#import lybraries
import sys
import os
import cifar10
sys.path.append('python')
import numpy as np
import ipdb
from fbCreate import fbCreate
from fbRun import fbRun
from assignTextons import assignTextons
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import plot_confusion_matrix 
from cifar10 import load_cifar10, get_data
import pickle
import random


testdata,testlabel=get_data(load_cifar10(mode='test'))
print (testdata.shape)
print(testdata[6759])

def listaAleatorios(n):
    lista = [0]  * n
    for i in range(n):
        lista[i] = random.randint(0, 10000)
    return lista

img=[]

for i in listaAleatorios(10):
    img.append(testdata[i])

def histc(X, bins):
    import numpy as np
    map_to_bins = np.digitize(X,bins)
    r = np.zeros(bins.shape)
    for i in map_to_bins:
        r[i-1] += 1
    return np.array(r)

fb = fbCreate(support=2, startSigma=0.6) 

textons_filename = "textons_exp8.pkl"  
with open(textons_filename, 'rb') as file:  
    textons = pickle.load(file)

clusters= 10

textonMap=[]
test_hist=[] 
for i in range (0,len(img)):
    tmap= assignTextons(fbRun(fb,img[i]),textons.transpose())
    textonMap.append(tmap)
    hist= histc(tmap.flatten(),np.arange(clusters))/tmap.size
    test_hist.append(hist)

model_filename = "model_exp8.pkl"  
with open(model_filename, 'rb') as file:  
    model = pickle.load(file)

prediction= model.predict(test_hist)
print('Hello Anaconda')

for nm in range(0,len(img)):
    plt.figure(1)
    plt.subplot(5,4,2*nm+1)
    plt.imshow(img[nm])
    plt.axis('off')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(15,15,prediction[nm],fontsize=11,bbox=props)
    plt.subplot(5,4,2*nm+2)
    plt.imshow(textonMap[nm])
    plt.axis('off')
    if nm == len(img)-1:
        plt.show()
