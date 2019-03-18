#!/usr/bin/ipython3

def save_cells(img_folder, K_vector):

    #import lybraries
    from segment_BSDS500 import segmentByClustering # Change this line if your function has a different name
    import imageio
    import os 
    import glob 
    import numpy as np
    import scipy.io as sio
    from cv2 import cv2
    from filtering import GHF
    import matplotlib.pyplot as plt

    img_path= os.path.join('BSR','BSDS500','data','images',img_folder,'*.jpg')
    images=glob.glob(img_path)
    folder_w= img_folder + '/watershed_cells'
    folder_k= img_folder +'/kmeans_cells_filter'
    if not os.path.exists(img_folder):
        os.mkdir(img_folder)
    if not os.path.exists(folder_k):
        os.mkdir(folder_k)
    if not os.path.exists(folder_w):
        os.mkdir(folder_w)
        
    for i in images:
        img = imageio.imread(i)
        
        Seg_watershed= np.zeros((len(K_vector)), dtype = np.object)
        Seg_kmeans= np.zeros((len(K_vector)), dtype = np.object)
        j=0
        for k in K_vector:   
            segmentation= segmentByClustering(img,'watershed',k)
            Seg_watershed[j]=segmentation
            j+=1

        img= GHF(i)
        j=0
        for k in K_vector:    
            segmentation= segmentByClustering(img,'kmeans',k)
            Seg_kmeans[j]=segmentation
            j+=1   

        name= i.split('/')
        name= name[5].split('.')
        name_watershed= folder_w + '/' + name[0] + '.mat'
        name_kmeans= folder_k + '/' + name[0] + '.mat'
        sio.savemat(name_watershed,{'segs':Seg_watershed})
        sio.savemat(name_kmeans,{'segs':Seg_kmeans})
        print (i)




