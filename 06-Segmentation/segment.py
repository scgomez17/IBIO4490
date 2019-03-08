#!/usr/bin/ipython3

def segmentByClustering(rgbImage, colorSpace, clusteringMethod, numberOfClusters):

    #import lybraries
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    from sklearn.cluster import AgglomerativeClustering
    from skimage.morphology import watershed
    from skimage.feature import peak_local_max
    from scipy import ndimage as ndi 
    from skimage import io, color
    import numpy as np
    from cv2 import cv2

    # Image colorspace processing
    if colorSpace== 'rgb':
        img = rgbImage
        rows, cols, dim= img.shape
        #Normalize the colorspace
        img = cv2.normalize(img,np.zeros((img.shape),dtype=np.uint8), alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    elif colorSpace=='lab':
        img = rgbImage
        rows, cols, dim= img.shape
        img = color.rgb2lab(img)
        #Normalize the colorspace
        img = cv2.normalize(img,np.zeros((img.shape),dtype=np.uint8), alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    elif colorSpace=='hsv':
        img = rgbImage
        rows, cols, dim= img.shape
        img = color.rgb2hsv(img)
        #Normalize the colorspace
        img = cv2.normalize(img,np.zeros((img.shape),dtype=np.uint8), alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    elif colorSpace=='rgbxy':
        img = rgbImage
        rows, cols, dim= img.shape
        x = np.zeros((rows,cols),dtype='uint8')
        y = np.zeros((rows,cols),dtype='uint8')
        for i in range(0,rows):
            for j in range(0,cols):
                x[i,:] = i
                y[:,j] = j
        #Normalize the colorspace
        img = cv2.normalize(img,np.zeros((img.shape),dtype=np.uint8), alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        x = cv2.normalize(x,np.zeros((x.shape),dtype=np.uint8), alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        y = cv2.normalize(y,np.zeros((y.shape),dtype=np.uint8), alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        img = np.dstack((img,x,y))    
        rows, cols, dim= img.shape
    elif colorSpace=='labxy':
        img = rgbImage
        img = color.rgb2lab(img)
        rows, cols, dim= img.shape
        x = np.zeros((rows,cols),dtype='uint8')
        y = np.zeros((rows,cols),dtype='uint8')
        for i in range(0,rows):
            for j in range(0,cols):
                x[i,:] = i
                y[:,j] = j
        #Normalize the colorspace
        img = cv2.normalize(img,np.zeros((img.shape),dtype=np.uint8), alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        x = cv2.normalize(x,np.zeros((x.shape),dtype=np.uint8), alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        y = cv2.normalize(y,np.zeros((y.shape),dtype=np.uint8), alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        img = np.dstack((img,x,y))   
        rows, cols, dim= img.shape 
    elif colorSpace=='hsvxy':
        img = rgbImage
        img = color.rgb2hsv(img)
        rows, cols, dim= img.shape
        x = np.zeros((rows,cols),dtype='uint8')
        y = np.zeros((rows,cols),dtype='uint8')
        for i in range(0,rows):
            for j in range(0,cols):
                x[i,:] = i
                y[:,j] = j
        #Normalize the colorspace
        img = cv2.normalize(img,np.zeros((img.shape),dtype=np.uint8), alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        x = cv2.normalize(x,np.zeros((x.shape),dtype=np.uint8), alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        y = cv2.normalize(y,np.zeros((y.shape),dtype=np.uint8), alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        img = np.dstack((img,x,y))   
        rows, cols, dim= img.shape 

    #Segmentation with a specif clustering method
    if clusteringMethod== 'kmeans':
        img= np.reshape(img,(rows*cols,dim))
        kmeans= KMeans(n_clusters=numberOfClusters,max_iter=162).fit_predict(img)
        segmentation= np.reshape(kmeans,(rows,cols))
    elif clusteringMethod=='gmm':
        img= np.reshape(img,(rows*cols,dim))
        gmm= GaussianMixture(n_components=numberOfClusters,covariance_type='full',max_iter=400).fit_predict(img)
        segmentation= np.reshape(gmm,(rows,cols))
    elif clusteringMethod== 'hierarchical':
        #Resize the image in order to decrease the processing time
        rows_resize= np.floor(rows/8)
        rows_resize= int(rows_resize)
        cols_resize= np.floor(cols/8)
        cols_resize= int(cols_resize)
        img= np.resize(img,(rows_resize,cols_resize,dim))
        img= np.reshape(img,(rows_resize*cols_resize,dim))
        hierarchical= AgglomerativeClustering(n_clusters=numberOfClusters,linkage='complete').fit_predict(img)
        segmentation= np.reshape(hierarchical,(rows_resize,cols_resize))
        segmentation= np.resize(segmentation,(rows,cols))
    elif clusteringMethod== 'watershed':
        img= np.floor(np.mean(img,axis=2))
        local_max= peak_local_max(-1*img,indices=False,num_peaks=numberOfClusters, num_peaks_per_label=1)
        markers= ndi.label(local_max)[0]
        watersheds= watershed(img, markers,compactness=1)
        segmentation= watersheds-1
    
    return segmentation


#sources
# https://scikit-learn.org/stable/modules/clustering.html
# http://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html