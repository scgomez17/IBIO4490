#!/usr/bin/ipython3

#Lybraries
import os 
import glob
import numpy as np
from obtain_labels_celeb import obtain_labels
import model_celeb
import dataset_celeb
import matplotlib.pyplot as plt
import random

#Torch lybraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
from tqdm import tqdm
import cv2

def AleatoryList(n):
    lista = [0]  * n
    for i in range(n):
        lista[i] = random.randint(0, 15000)
    return lista

def demo ():
    
    model = model_celeb.Net().to(device)
    model.load_state_dict(torch.load('model.pth','cpu'))
    model.eval()

    img_path= os.path.join(path,'img_align_celeba','*.jpg')
    images=sorted(glob.glob(img_path))
    images= images[182638:len(images)]
    
    cascPath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)

    plt.figure()
    for i in AleatoryList(8):

        img = cv2.imread(images[i])
        plt.subplot(2,4,i+1)
        plt.imshow(img)
        plt.axis('off')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
        faces = faceCascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces)>=1:
            img= img[faces[0][1]:faces[0][1]+faces[0][3],faces[0][0]:faces[0][0]+faces[0][2]]
            img= cv2.resize(img,(int(64),int(64)))
        else:
            img= cv2.resize(img,(int(64),int(64)))
        img = cv2.resize(img,(int(64),int(64)))
        img= torch.stack([torch.Tensor(img)])
        img = Variable(img).unsqueeze(0).cuda(1)
        
        output= model (img)
        output = torch.sigmoid(output)
        pred= torch.round(output.data.cpu())        
        pred= pred.numpy()
        ann= str(int(pred[0][0]))+','+str(int(pred[0][1]))+','+str(int(pred[0][2]))+','+str(int(pred[0][3]))+','+str(int(pred[0][4]))+','+str(int(pred[0][5])) +','+str(int(pred[0][6]))+','+str(int(pred[0][7]))+','+str(int(pred[0][8]))+','+str(int(pred[0][9]))

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(125,125,ann,fontsize=20,bbox=props)
    
    plt.savefig('Demo.png')
    plt.show()
    
if __name__=='__main__':
    #Database Path
    folder= os.path.isdir('CelebA')

    if folder==True:
        path = 'CelebA/'
    else:
        path = '/media/user_home2/vision/data/CelebA/'
    
    device=torch.device('cuda:1' if torch.cuda.is_available() else "cpu")
    print (device)
    batch_size=50
    demo () 



