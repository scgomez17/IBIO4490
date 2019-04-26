import os
import numpy as np 
from torch.utils.data import Dataset, DataLoader
from obtain_labels_celeb import obtain_labels
import cv2
import glob
import torch

class Data(Dataset):
    def __init__(self,path,test = False,val = False):
    
        img_path= os.path.join(path,'img_align_celeba','*.jpg')
        images=sorted(glob.glob(img_path))

        labels= obtain_labels(path)
        if val:
            labels=labels[162770:182638]
            images= images[162770:182638]
        elif test:
            images= images[182638:len(images)]
        else:
            labels= labels[0:162770]
            labels=labels[0:3000]
            images= images[0:162770]
            images= images[0:3000]
            
        ind=0
        cascPath = "haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(cascPath)
        for i in range(len(images)):
            images[i]= cv2.imread(images[i])
            images[i]= cv2.cvtColor(images[i],cv2.COLOR_BGR2GRAY) 
            faces = faceCascade.detectMultiScale(images[i], scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            if len(faces)>=1:
                images[i]= images[i][faces[0][1]:faces[0][1]+faces[0][3],faces[0][0]:faces[0][0]+faces[0][2]]
                images[i]= cv2.resize(images[i],(int(64),int(64)))
            else:
                images[i]= cv2.resize(images[i],(int(64),int(64)))
            print (ind)
            ind +=1

        images=np.array(images[:])
        images = images[:, np.newaxis]
        images=torch.stack([torch.Tensor(i) for i in images])

        labels= torch.stack([torch.Tensor(i) for i in labels])
        self.image_files=images
        self.labels=labels
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self,idx):
        return (self.image_files[idx], self.labels[idx])

