#!/usr/bin/python3

#import lybraries
import os 
import numpy as np
from PIL import Image,ImageFilter
import matplotlib.pyplot as plt
import cv2

#Pathways of the original images
p_cam= os.path.join ('cam.jpg')
p_dan= os.path.join ('dan.jpg')

#Resize the original images and save them 
size= (512,512)
img_cam= Image.open(p_cam)
img_cam= img_cam.resize (size,Image.ANTIALIAS)
img_cam.save("md_cam.jpg")
img_dan= Image.open(p_dan)
img_dan= img_dan.resize(size,Image.ANTIALIAS)
img_dan.save("md_dan.jpg")

#Pathways of the modified images
p_cam= os.path.join ('md_cam.jpg')
p_dan= os.path.join ('md_dan.jpg')

#Filtering and showing the results
img_dan= cv2.imread(p_dan)
img_dan=cv2.cvtColor(img_dan,cv2.COLOR_BGR2RGB)
f_dan=cv2.GaussianBlur(img_dan,(55,55),9)

fig=plt.figure()
plt.imshow(img_dan)
plt.show()

fig = plt.figure()
plt.imshow(f_dan)
plt.title('Low pass filter')
plt.show()

img_cam= cv2.imread(p_cam)
gs_cam=cv2.cvtColor(img_cam,cv2.COLOR_BGR2GRAY)
g_cam=cv2.GaussianBlur(gs_cam,(55,55),3)
f_cam=gs_cam-g_cam
f_cam=np.stack((f_cam,f_cam,f_cam),axis=-1)

fig=plt.figure()
plt.imshow(img_cam)
plt.show()

plt.show()
fig=plt.figure()
plt.imshow(gs_cam)

fig = plt.figure()
plt.imshow(f_cam)
plt.title('High pass filter')
plt.show()


#Hybrid image
Hybrid = f_dan+f_cam
fig = plt.figure()
plt.imshow(Hybrid)
plt.axis('off')
plt.title('Hybrid image')
plt.show()


#Generate gaussian pyramid for hybrid image
GP_Hybrid= [Hybrid]

for i in range(1,6):
     GP = cv2.pyrDown(GP_Hybrid[i-1])
     GP_Hybrid.append(GP)

#Show Hybrid gaussian pyramid
rows,cols,_= Hybrid.shape
pyramid = np.zeros((rows, cols + cols // 2, 3), dtype=np.uint8)
pyramid[:rows,:cols,:]= Hybrid
rows_p=0
for i in range(1,6):
        pyramid_img=GP_Hybrid[i]
        rows_img, cols_img,_= pyramid_img.shape
        pyramid[rows_p:rows_p+rows_img, cols:cols + cols_img]= pyramid_img
        rows_p += rows_img

plt.imshow(pyramid)
plt.axis('off')
plt.title('Hybrid image Gaussian Pyramid')
plt.show()

