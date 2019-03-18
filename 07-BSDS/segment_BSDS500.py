#!/usr/bin/ipython3

def segmentByClustering(rgbImage, clusteringMethod, numberOfClusters):

    #import lybraries
    from sklearn.cluster import KMeans   
    from skimage.morphology import watershed
    from skimage.feature import peak_local_max
    from scipy import ndimage as ndi 
    import numpy as np

    img=rgbImage
    rows, cols, dim= rgbImage.shape
    #Segmentation with a specif clustering method
    if clusteringMethod== 'kmeans':
        img= np.reshape(img,(rows*cols,dim))
        kmeans= KMeans(n_clusters=numberOfClusters,max_iter=162).fit_predict(img)
        segmentation= np.reshape(kmeans,(rows,cols))
        segmentation= segmentation +1
    elif clusteringMethod== 'watershed':
        img= np.floor(np.mean(img,axis=2))
        local_max= peak_local_max(-1*img,indices=False,num_peaks=numberOfClusters, num_peaks_per_label=1)
        markers= ndi.label(local_max)[0]
        watersheds= watershed(img, markers,compactness=1)
        segmentation= watersheds
    
    return segmentation


#sources
# https://scikit-learn.org/stable/modules/clustering.html
# http://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html
