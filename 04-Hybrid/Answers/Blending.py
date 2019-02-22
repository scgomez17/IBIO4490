#!/usr/bin/python3

#import lybraries
import os 
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np

## Messi and CR7

#Path of the original images
Path_1= os.path.join ('dan.jpg')
Path_2= os.path.join ('cam.jpg')

#Resize the original images and save them 
size= (512,512)
Img_1= Image.open(Path_1)
Img_1= Img_1.resize (size,Image.ANTIALIAS)
Img_1.save("md_dan.jpg")
Img_2= Image.open(Path_2)
Img_2= Img_2.resize(size,Image.ANTIALIAS)
Img_2.save("md_cam.jpg")

#Read the images and convert them to rgb
Img_1= cv2.imread("md_dan.jpg")
Img_1 = cv2.cvtColor(Img_1,cv2.COLOR_BGR2RGB)
Img_2 = cv2.imread("md_cam.jpg")
Img_2 = cv2.cvtColor(Img_2,cv2.COLOR_BGR2RGB)


#Generate gaussian and laplacian pyramid for the two images. 
GP_Img_1= [Img_1]
GP_Img_2 = [Img_2]
LP_Img_1= []
LP_Img_2 = []

for i in range(1,6):
     GP = cv2.pyrDown(GP_Img_1[i-1])
     GP_Img_1.append(GP)
     GP = cv2.pyrDown(GP_Img_2[i-1])
     GP_Img_2.append(GP)
     LP = cv2.pyrUp(GP_Img_1[i])
     L = cv2.subtract(GP_Img_1[i-1],LP)
     LP_Img_1.append(L)
     LP = cv2.pyrUp(GP_Img_2[i])
     L = cv2.subtract(GP_Img_2[i-1],LP)
     LP_Img_2.append(L)

LP_Img_1.append(GP_Img_1[5])
LP_Img_2.append(GP_Img_2[5])

#Show Img_1's gaussian pyramid
rows,cols,_= Img_1.shape
pyramid = np.zeros((rows, cols + cols // 2, 3), dtype=np.uint8)
pyramid[:rows,:cols,:]= Img_1
rows_p=0
for i in range(1,6):
        pyramid_img=GP_Img_1[i]
        rows_img, cols_img,_= pyramid_img.shape
        pyramid[rows_p:rows_p+rows_img, cols:cols + cols_img]= pyramid_img
        rows_p += rows_img

plt.imshow(pyramid)
plt.axis('off')
plt.title('Image 1 Gaussian Pyramid')
plt.show()

#Show Img_2's gaussian pyramid
rows,cols,_= Img_2.shape
pyramid = np.zeros((rows, cols + cols // 2, 3), dtype=np.uint8)
pyramid[:rows,:cols,:]= Img_2
rows_p=0
for i in range(1,6):
        pyramid_img=GP_Img_2[i]
        rows_img, cols_img,_= pyramid_img.shape
        pyramid[rows_p:rows_p+rows_img, cols:cols + cols_img]= pyramid_img
        rows_p += rows_img

plt.imshow(pyramid)
plt.axis('off')
plt.title('Image 2 Gaussian Pyramid')
plt.show()

#Show Img_1's laplacian pyramid
rows,cols,_= LP_Img_1[0].shape
pyramid = np.zeros((rows, cols + cols // 2, 3), dtype=np.uint8)
pyramid[:rows,:cols,:]= LP_Img_1[0]
rows_p=0
for i in range(1,5):
        pyramid_img=LP_Img_1[i]
        rows_img, cols_img,_= pyramid_img.shape
        pyramid[rows_p:rows_p+rows_img, cols:cols + cols_img]= pyramid_img
        rows_p += rows_img

plt.imshow(pyramid)
plt.axis('off')
plt.title('Image 1 Laplacian Pyramid')
plt.show()

#Show Img_2's laplacian pyramid
rows,cols,_= LP_Img_2[0].shape
pyramid = np.zeros((rows, cols + cols // 2, 3), dtype=np.uint8)
pyramid[:rows,:cols,:]= LP_Img_2[0]
rows_p=0
for i in range(1,5):
        pyramid_img=LP_Img_2[i]
        rows_img, cols_img,_= pyramid_img.shape
        pyramid[rows_p:rows_p+rows_img, cols:cols + cols_img]= pyramid_img
        rows_p += rows_img

plt.imshow(pyramid)
plt.axis('off')
plt.title('Image 2 Laplacian Pyramid')
plt.show()


#Join left part of Img_1 and rigth part of Img_2 in a new image for each level. 
#This method also do the reconstruction using laplacian pyramid and the 
#high level of gaussian pyramid
Blending= []
f=0
for i in range(5,-1,-1):
    img1=LP_Img_1[i]
    img2=LP_Img_2[i]
    _,cols,_= img2.shape 
    B= np.hstack((img1[:,0:int(cols/2),:],img2[:,int(cols/2):,:]))
    Blending.append(B)

    if i == 4:
        Blending_UP = cv2.pyrUp(Blending[f])
        Blending_UP = cv2.add(Blending_UP, Blending[f+1])
        f=f+1
    elif i<4:
        Blending_UP = cv2.pyrUp(Blending_UP)
        Blending_UP = cv2.add(Blending_UP, Blending[f+1])   
        f=f+1
        
plt.imshow(Blending_UP)
plt.axis('off')
plt.title('Blending image using pyramids')
plt.show()


#Generate gaussian pyramid for blending image
GP_Blending= [Blending_UP]

for i in range(1,6):
     GP = cv2.pyrDown(GP_Blending[i-1])
     GP_Blending.append(GP)

#Show Blending gaussian pyramid
rows,cols,_= Blending_UP.shape
pyramid = np.zeros((rows, cols + cols // 2, 3), dtype=np.uint8)
pyramid[:rows,:cols,:]= Blending_UP
rows_p=0
for i in range(1,6):
        pyramid_img=GP_Blending[i]
        rows_img, cols_img,_= pyramid_img.shape
        pyramid[rows_p:rows_p+rows_img, cols:cols + cols_img]= pyramid_img
        rows_p += rows_img

plt.imshow(pyramid)
plt.axis('off')
plt.title('Blending image Gaussian Pyramid')
plt.show()

#Image without pyramid blending 
_,cols,_= Img_1.shape 
Blending_WP= np.hstack((Img_1[:,0:int(cols/2),:],Img_2[:,int(cols/2):,:]))
plt.imshow(Blending_WP)
plt.axis('off')
plt.title('Blending image without pyramids')
plt.show()

#Sources
#https://docs.opencv.org/3.1.0/dc/dff/tutorial_py_pyramids.html
#https://docs.scipy.org/doc/numpy/reference/generated/numpy.hstack.html
#http://scikit-image.org/docs/dev/auto_examples/transform/plot_pyramid.html

