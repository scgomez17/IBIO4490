#!/usr/bin/ipython3

#import lybraries
import sys
import os
import cifar10
sys.path.append('python')
import numpy as np
import ipdb
from fbCreate import fbCreate
from fbRun import fbRun
from assignTextons import assignTextons
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import plot_confusion_matrix 
from cifar10 import load_cifar10, get_data
import pickle


testdata,testlabel=get_data(load_cifar10(mode='test'))
print (testdata.shape)

img_test=testdata[0]

#Concatenate images
for i in range(1,len(testdata)):
    img_test=np.hstack((img_test,testdata[i]))


#Function to obtain the histogram of textons
def histc(X, bins):
    import numpy as np
    map_to_bins = np.digitize(X,bins)
    r = np.zeros(bins.shape)
    for i in map_to_bins:
        r[i-1] += 1
    return np.array(r)

#Create filter bank
fb = fbCreate(support=2, startSigma=0.6) 

#load textons 
textons_filename = "textons_new.pkl"  
with open(textons_filename, 'rb') as file:  
    textons = pickle.load(file)

clusters= 16*16 #optimum clusters

# Calculate textons representation and assign them
test_hist=[] 
for i in range (0,10000):
    tmap= assignTextons(fbRun(fb,testdata[i]),textons.transpose())
    hist= histc(tmap.flatten(),np.arange(clusters))/tmap.size
    test_hist.append(hist)

#Predict labels of the test set
model_filename = "model_new.pkl"  
with open(model_filename, 'rb') as file:  
    model = pickle.load(file)

prediction= model.predict(test_hist)

ACA= accuracy_score(testlabel,prediction)
CM=confusion_matrix(testlabel,prediction)

posible_labels= range(0,10)
plt.figure()
plot_confusion_matrix.plot_confusion_matrix(CM,posible_labels,normalize=True,title='Normalize confusion matrix')
plt.savefig('Confusion_matrix_test_exp17.png')
plt.show()

print ('ACA measure is ' + str(ACA))

#source
#https://stackabuse.com/scikit-learn-save-and-restore-models/
