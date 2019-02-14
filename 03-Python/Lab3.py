#!/usr/bin/python3
# import the modules needed to run the script
import os
import time
import zipfile
import urllib.request
import glob
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import shutil
# Start processing time
start_time = time.process_time()
# First, it is check that the dataset isn't downloaded alredy and download the dataset from Dropbox
# The dataset was taken from Real and Fake Face Detection availible at: https://www.kaggle.com/ciplab/real-and-fake-face-detection#real_and_fake_face_detection.zip
# A part of the original dataset was used
unzip_path=os.path.isdir('faces.zip')
zip_path=os.path.isfile('faces')
if (unzip_path==True and zip_path==False):
    d=zipfile.ZipFile('faces.zip','r')
    d.extractall()
    d.close()
elif (unzip_path==False and zip_path==False):
    url='https://www.dropbox.com/s/sd243eckykxqf7y/faces.zip?dl=1'
    resp=urllib.request.urlopen(url)
    urllib.request.urlretrieve(url,'faces.zip')
    d=zipfile.ZipFile('faces.zip','r')
    d.extractall()
    d.close()
# Create a new folder to store 8 randomly selected images from the original dataset
if not os.path.exists('Files'):
    os.makedirs('Files')
# Define a function to create a list with random numbers
def listaAleatorios(n):
      lista = [0]  * n
      for i in range(n):
          lista[i] = random.randint(0, 400)
      return lista
# Save in a list the pathname of the images downloaded
orig_li=glob.glob(os.path.join('faces','*','*.jpg'))
# Iterate the list with pathnames taking the images that match with the random index list created
# resize to 256x256 and save them in the new folder created
for name in listaAleatorios(8):
    img=Image.open(orig_li[name])
    fn= os.path.basename(orig_li[name])
    new_img=img.resize((256,256))
    new_img.save('Files/'+fn+'.jpg','JPEG')
# Create a sub-list for the images randomly selected
sub_li=glob.glob(os.path.join('Files','*.jpg'))
# Iterate to read the images and subplot them. Additonally, since the labels are in the pathname of the image
# the pathname is splited. If an image is 'real' the label will be 1, else the label will be 0.
for nm in range(0,8):
    plt.figure(1)
    plt.subplot(2,4,nm+1)
    plt.imshow(plt.imread(sub_li[nm]))
    plt.axis('off')
    s=os.path.basename(sub_li[nm])
    a = s.split('_')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    if (a[0]=='real'):
        plt.text(125,125,'1',fontsize=20,bbox=props)
    else:
        plt.text(125,125,'0',fontsize=20,bbox=props)
    if nm == 7:
        plt.show()
# Finally, the folder created is deleted and the processing time showed in command window
shutil.rmtree('Files')
print (time.process_time() - start_time, "seconds")