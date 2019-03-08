#!/usr/bin/ipython3

def indJaccard (seg,gt):  

    #import lybraries
    from cv2 import cv2 
    import numpy as np

    #Normalization
    seg=seg+1
    seg= cv2.normalize(seg,np.zeros((seg.shape),dtype=np.uint8), alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    gt= cv2.normalize(gt,np.zeros((gt.shape),dtype=np.uint8), alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    #Jaccard index measure
    inter = gt & seg #intersection
    union = gt | seg #Union
    metric= inter.sum()/union.sum()
    return metric
