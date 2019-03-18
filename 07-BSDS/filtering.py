
#!/usr/bin/ipython3

#Gaussian hierarchical filtering
def GHF(path):

    #import lybraries
    from cv2 import cv2
    import numpy as np
    import imageio
    import matplotlib.pyplot as plt

    img = imageio.imread(path)
    img= cv2.GaussianBlur(img,(7,7),0,0)
    big= imageio.imread(path)
    big= cv2.resize(big,None,fx=2,fy=2)
    big= cv2.GaussianBlur(big,(7,7),0,0)
    big= cv2.resize(big,None,fx=0.5,fy=0.5)

    c1= np.dstack((img[:,:,0],big[:,:,0]))
    c1= np.floor(np.mean(c1,axis=2))
    c2= np.dstack((img[:,:,1],big[:,:,1]))
    c2= np.floor(np.mean(c2,axis=2))
    c3= np.dstack((img[:,:,2],big[:,:,2]))
    c3= np.floor(np.mean(c3,axis=2))

    final_img= np.stack((c1,c2,c3), axis=-1)
    final_img = final_img.astype(np.uint8)
    
    return final_img


