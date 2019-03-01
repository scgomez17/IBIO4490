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
from computeTextons import computeTextons
from assignTextons import assignTextons
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import plot_confusion_matrix 
from cifar10 import load_cifar10, get_data
import pickle
import timeit

#Get the data and labels of the train
data=dict()
traindata=dict()
trainlabel=dict()
for i in range(1,6):
    data[i-1]=cifar10.load_cifar10(meta='cifar-10-batches-py',mode=i)
    traindata[i-1],trainlabel[i-1]=cifar10.get_data(data[i-1])
    trainlabel[i-1]=trainlabel[i-1][0:10000]

images= np.concatenate((traindata[0],traindata[1],traindata[2],traindata[3],traindata[4]))
labels= np.concatenate((trainlabel[0],trainlabel[1],trainlabel[2],trainlabel[3],trainlabel[4]))

index= []
ind= 30
for i in range (0,10):
    ls= np.where(labels==i)
    ls= ls[0][0:ind]
    index.append(ls)

img= []
l=[]
for i in range (0,10):
    for j in index[i]:
        img.append(images[j])
        l.append(labels[j])

img_train=img[0]
label_train= l[0]
#Concatenate images
for i in range(1,len(img)):
    img_train=np.hstack((img_train,img[i]))
    label_train= np.hstack((label_train,l[i]))

#Create filter bank
fb = fbCreate(support=2, startSigma=0.6) 

#Function to obtain the histogram of textons
def histc(X, bins):
    import numpy as np
    map_to_bins = np.digitize(X,bins)
    r = np.zeros(bins.shape)
    for i in map_to_bins:
        r[i-1] += 1
    return np.array(r)

##Changing K to choose the optimum K. The number of images per each batch is constant. KNN as classifier KNN as classifier 
filterResponses = fbRun(fb,img_train)

ACA_K=0
clusters= 0 #optimum clusters
K= range (2,150)
ACAs= []
# Compute, calculate, assign textons and
for k in K:
        #C ompute textons from filter images
        
        map, textons = computeTextons(filterResponses,k)

        # Calculate textons representation and assign them
        train_hist=[] 
        for i in range (0,10*ind):
                tmap= assignTextons(fbRun(fb,img[i]),textons.transpose())
                hist= histc(tmap.flatten(),np.arange(k))/tmap.size
                train_hist.append(hist)

        # Get model, classify and ACA
        model= KNeighborsClassifier(n_neighbors=3) 
        model= model.fit(train_hist,l) 
        prediction= model.predict(train_hist)
        ACA_aux= accuracy_score(l,prediction)
        ACAs.append(ACA_aux)
        # Selection of the optimum K
        if ACA_aux>ACA_K:
                ACA_K= ACA_aux
                clusters=k
                CM=confusion_matrix(l,prediction)

posible_labels= range(0,10)
plt.figure()
plot_confusion_matrix.plot_confusion_matrix(CM,posible_labels,normalize=True,title='Normalize confusion matrix')
plt.show()

plt.figure()
plt.plot(K,ACAs)
plt.xlabel('Number of clusters (K)')
plt.ylabel('ACA')
plt.savefig('ACA_vs_K.png')
plt.show()

print ('the number of clusters is ' + str(clusters)) 

#####################################################################################################################

##Changing the number of the images using the K optimum found above. KNN as classifier 

ACA_img=0
images_cat=0
ACAs= []
IND= range (2,100)
for ind in IND:
    index= []

    for i in range (0,10):
        ls= np.where(labels==i)
        ls= ls[0][0:ind]
        index.append(ls)

    img= []
    l=[]
    for i in range (0,10):
        for j in index[i]:
            img.append(images[j])
            l.append(labels[j])

    img_train=img[0]
    label_train= l[0]
    #Concatenate images
    for i in range(1,len(img)):
        img_train=np.hstack((img_train,img[i]))
        label_train= np.hstack((label_train,l[i]))

    filterResponses = fbRun(fb,img_train)

    #C ompute textons from filter images
        
    map, textons = computeTextons(filterResponses,clusters)

    # Calculate textons representation and assign them
    train_hist=[] 
    for i in range (0,10*ind):
            tmap= assignTextons(fbRun(fb,img[i]),textons.transpose())
            hist= histc(tmap.flatten(),np.arange(clusters))/tmap.size
            train_hist.append(hist)

    # Get model, classify and ACA
    model= KNeighborsClassifier(n_neighbors=3) 
    model= model.fit(train_hist,l) 
    prediction= model.predict(train_hist)
    ACA_aux= accuracy_score(l,prediction)
    ACAs.append(ACA_aux)
    # Selection of the optimum numbers of images
    if ACA_aux>ACA_img:
        ACA_img= ACA_aux
        images_cat=ind
        CM=confusion_matrix(l,prediction)

plt.figure()
plot_confusion_matrix.plot_confusion_matrix(CM,posible_labels,normalize=True,title='Normalize confusion matrix')
plt.show()

plt.figure()
plt.plot(IND,ACAs)
plt.xlabel('Number of images per class')
plt.ylabel('ACA')
plt.title('ACA vs number of images per clas')
plt.show()

print ('The number of images per class is ' + str(images_cat))
            
#####################################################################################################################

## changing neighbors of KNN and trees of random forest. Use K and the number of images per class 

index= []
ind= images_cat
for i in range (0,10):
    ls= np.where(labels==i)
    ls= ls[0][0:ind]
    index.append(ls)

img= []
l=[]
for i in range (0,10):
    for j in index[i]:
        img.append(images[j])
        l.append(labels[j])

img_train=img[0]
label_train= l[0]
#Concatenate images
for i in range(1,len(img)):
    img_train=np.hstack((img_train,img[i]))
    label_train= np.hstack((label_train,l[i]))

##Changing K to choose the optimum K. The number of images per each batch is constant 
filterResponses = fbRun(fb,img_train)

#Compute textons from filter images       
map, textons = computeTextons(filterResponses,clusters)

# Calculate textons representation and assign them
train_hist=[] 
for i in range (0,10*ind):
        tmap= assignTextons(fbRun(fb,img[i]),textons.transpose())
        hist= histc(tmap.flatten(),np.arange(clusters))/tmap.size
        train_hist.append(hist)

# Get model, classify with KNN and ACA
ACA_KNN=0
neighbors=0
Neig=range (2,100)
ACAs= []
for i in Neig:
    model= KNeighborsClassifier(n_neighbors=i) 
    model= model.fit(train_hist,l) 
    prediction= model.predict(train_hist)
    ACA_aux= accuracy_score(l,prediction)
    ACAs.append(ACA_aux)
    # Selection of the optimum K
    if ACA_aux>ACA_KNN:
        ACA_KNN= ACA_aux
        neighbors=i
        CM=confusion_matrix(l,prediction)

plt.figure()
plot_confusion_matrix.plot_confusion_matrix(CM,posible_labels,normalize=True,title='Normalize confusion matrix with KNN')
plt.show()

plt.figure()
plt.plot(Neig,ACAs)
plt.xlabel('Number of neighbors')
plt.ylabel('ACA')
plt.title('ACA vs number of neighbors')
plt.show()
print ('The number of neighbors is ' + str(neighbors))


# Get model, classify with Random forest and ACA
ACA_Random_Forest=0
Forest=range (2,100)
ACAs= []
forest=0
for i in Forest:
    model= RandomForestClassifier(n_estimators=i)
    model= model.fit(train_hist,l) 
    prediction= model.predict(train_hist)
    ACA_aux= accuracy_score(l,prediction)
    ACAs.append(ACA_aux)
    # Selection of the optimum K
    if ACA_aux>ACA_Random_Forest:
        ACA_Random_Forest= ACA_aux
        forest=i
        CM=confusion_matrix(l,prediction)

plt.figure()
plot_confusion_matrix.plot_confusion_matrix(CM,posible_labels,normalize=True,title='Normalize confusion matrix with Random forest')
plt.show()

plt.figure()
plt.plot(Neig,ACAs)
plt.xlabel('Number of trees')
plt.ylabel('ACA')
plt.title('ACA vs number of trees')
plt.show()
print ('The number os trees is ' + str(forest))

#####################################################################################################################

##Changing K to choose the optimum K. Random forest as classifier

index= []
ind= images_cat
for i in range (0,10):
    ls= np.where(labels==i)
    ls= ls[0][0:ind]
    index.append(ls)

img= []
l=[]
for i in range (0,10):
    for j in index[i]:
        img.append(images[j])
        l.append(labels[j])

img_train=img[0]
label_train= l[0]
#Concatenate images
for i in range(1,len(img)):
    img_train=np.hstack((img_train,img[i]))
    label_train= np.hstack((label_train,l[i]))

#Create filter bank
fb = fbCreate(support=2, startSigma=0.6) 

filterResponses = fbRun(fb,img_train)

ACA_K=0
clusters= 0 #optimum clusters
K= range (2,100)
ACAs= []
# Compute, calculate, assign textons and
for k in K:
        #C ompute textons from filter images
        
        map, textons = computeTextons(filterResponses,k)

        # Calculate textons representation and assign them
        train_hist=[] 
        for i in range (0,10*ind):
                tmap= assignTextons(fbRun(fb,img[i]),textons.transpose())
                hist= histc(tmap.flatten(),np.arange(k))/tmap.size
                train_hist.append(hist)

        # Get model, classify and ACA
        model= RandomForestClassifier(n_estimators=forest)
        model= model.fit(train_hist,l) 
        prediction= model.predict(train_hist)
        ACA_aux= accuracy_score(l,prediction)
        ACAs.append(ACA_aux)
        # Selection of the optimum K
        if ACA_aux>ACA_K:
                ACA_K= ACA_aux
                clusters=k
                CM=confusion_matrix(l,prediction)

plt.figure()
plot_confusion_matrix.plot_confusion_matrix(CM,posible_labels,normalize=True,title='Normalize confusion matrix using Random Forest')
plt.show()

plt.figure()
plt.plot(K,ACAs)
plt.xlabel('Number of clusters (K)')
plt.ylabel('ACA')
plt.title('ACA vs number of clusters')
plt.show()

print ('the number of clusters is ' + str(clusters)) 

#####################################################################################################################


##Changing the number of the images using the K optimum found above. Random forest as classifier 

ACA_img=0
images_cat=0
ACAs= []
IND= range (2,100)
for ind in IND:
    index= []

    for i in range (0,10):
        ls= np.where(labels==i)
        ls= ls[0][0:ind]
        index.append(ls)

    img= []
    l=[]
    for i in range (0,10):
        for j in index[i]:
            img.append(images[j])
            l.append(labels[j])

    img_train=img[0]
    label_train= l[0]
    #Concatenate images
    for i in range(1,len(img)):
        img_train=np.hstack((img_train,img[i]))
        label_train= np.hstack((label_train,l[i]))

    filterResponses = fbRun(fb,img_train)

    #C ompute textons from filter images
        
    map, textons = computeTextons(filterResponses,clusters)

    # Calculate textons representation and assign them
    train_hist=[] 
    for i in range (0,10*ind):
            tmap= assignTextons(fbRun(fb,img[i]),textons.transpose())
            hist= histc(tmap.flatten(),np.arange(clusters))/tmap.size
            train_hist.append(hist)

    # Get model, classify and ACA
    model= KNeighborsClassifier(n_neighbors=3) 
    model= model.fit(train_hist,l) 
    prediction= model.predict(train_hist)
    ACA_aux= accuracy_score(l,prediction)
    ACAs.append(ACA_aux)
    # Selection of the optimum numbers of images
    if ACA_aux>ACA_img:
        ACA_img= ACA_aux
        images_cat=ind
        CM=confusion_matrix(l,prediction)

plt.figure()
plot_confusion_matrix.plot_confusion_matrix(CM,posible_labels,normalize=True,title='Normalize confusion matrix using Random Forest')
plt.show()

plt.figure()
plt.plot(IND,ACAs)
plt.xlabel('Number of images per class')
plt.ylabel('ACA')
plt.title('ACA vs number of images per clas')
plt.show()

print ('The number of images per class is ' + str(images_cat))


#####################################################################################################################

## changing trees of random forest. Use K and the number of images per class 

index= []
ind= images_cat
for i in range (0,10):
    ls= np.where(labels==i)
    ls= ls[0][0:ind]
    index.append(ls)

img= []
l=[]
for i in range (0,10):
    for j in index[i]:
        img.append(images[j])
        l.append(labels[j])

img_train=img[0]
label_train= l[0]
#Concatenate images
for i in range(1,len(img)):
    img_train=np.hstack((img_train,img[i]))
    label_train= np.hstack((label_train,l[i]))

##Changing K to choose the optimum K. The number of images per each batch is constant 
filterResponses = fbRun(fb,img_train)

#Compute textons from filter images       
map, textons = computeTextons(filterResponses,clusters)

# Calculate textons representation and assign them
train_hist=[] 
for i in range (0,10*ind):
        tmap= assignTextons(fbRun(fb,img[i]),textons.transpose())
        hist= histc(tmap.flatten(),np.arange(clusters))/tmap.size
        train_hist.append(hist)

# Get model, classify with Random forest and ACA
ACA_Random_Forest=0
Forest=range (2,200)
ACAs= []
forest=0
for i in Forest:
    model= RandomForestClassifier(n_estimators=i)
    model= model.fit(train_hist,l) 
    prediction= model.predict(train_hist)
    ACA_aux= accuracy_score(l,prediction)
    ACAs.append(ACA_aux)
    # Selection of the optimum K
    if ACA_aux>ACA_Random_Forest:
        ACA_Random_Forest= ACA_aux
        forest=i
        CM=confusion_matrix(l,prediction)

plt.figure()
plot_confusion_matrix.plot_confusion_matrix(CM,posible_labels,normalize=True,title='Normalize confusion matrix with Random forest')
plt.show()

plt.figure()
plt.plot(Neig,ACAs)
plt.xlabel('Number of trees')
plt.ylabel('ACA')
plt.title('ACA vs number of trees')
plt.show()
print ('The number os trees is ' + str(forest))


#####################################################################################################################

##Model used

index= []
ind= 50 #optimum number of images
for i in range (0,10):
    ls= np.where(labels==i)
    ls= ls[0][0:ind]
    index.append(ls)

img= []
l=[]
for i in range (0,10):
    for j in index[i]:
        img.append(images[j])
        l.append(labels[j])

img_train=img[0]
label_train= l[0]
#Concatenate images
for i in range(1,len(img)):
    img_train=np.hstack((img_train,img[i]))
    label_train= np.hstack((label_train,l[i]))

#Create filter bank
fb = fbCreate(support=2, startSigma=0.6) 

filterResponses = fbRun(fb,img_train)

clusters= 16*16 #optimum clusters

tic= timeit.default_timer()
#Compute textons from filter images       
map, textons = computeTextons(filterResponses,clusters)
toc= timeit.default_timer()
process_time= toc - tic
print ('The time to create the textons dictionary is ' + str(process_time))

#Save textons 
textons_filename = "textons_new.pkl"  
with open(textons_filename, 'wb') as file:  
    pickle.dump(textons, file)


# Calculate textons representation and assign them
train_hist=[] 
for i in range (0,10*ind):
        tmap= assignTextons(fbRun(fb,img[i]),textons.transpose())
        hist= histc(tmap.flatten(),np.arange(clusters))/tmap.size
        train_hist.append(hist)
        

# Get model, classify with Random forest and ACA
ACA_Random_Forest=0
forest= 175#optimum number of trees
ACAs= []
forest=0
for i in (0,100):
    model= RandomForestClassifier(n_estimators=forest)
    model= model.fit(train_hist,l) 
    prediction= model.predict(train_hist)
    ACA_aux= accuracy_score(l,prediction)
    ACAs.append(ACA_aux)
    # Selection of the optimum K
    if ACA_aux>ACA_Random_Forest:
        ACA_Random_Forest= ACA_aux
        forest=i
        CM=confusion_matrix(l,prediction)
        model_filename = "model_new.pkl"  
        with open(model_filename, 'wb') as file:  
            pickle.dump(model, file)

plt.figure()
plot_confusion_matrix.plot_confusion_matrix(CM,posible_labels,normalize=True,title='Normalize confusion matrix final model')
plt.show()

print ('Aca measure is ' + str(ACA_Random_Forest))       


#####################################################################################################################

## Test


testdata,testlabel=get_data(load_cifar10(mode='test'))

images= np.concatenate((testdata[0],testdata[1],testdata[2],testdata[3],testdata[4]))
labels= np.concatenate((testlabel[0],testlabel[1],testlabel[2],testlabel[3],testlabel[4]))

index= []
ind= 10000
for i in range (0,10):
    ls= np.where(labels==i)
    ls= ls[0][0:ind]
    index.append(ls)

img= []
l=[]
for i in range (0,10):
    for j in index[i]:
        img.append(images[j])
        l.append(labels[j])

img_test=img[0]
label_test= l[0]

#Concatenate images
for i in range(1,len(img)):
    img_test=np.hstack((img_test,img[i]))
    label_test= np.hstack((label_test,l[i]))

#Create filter bank
fb = fbCreate(support=2, startSigma=0.6) 

#load textons 
textons_filename = "textons_new.pkl"  
with open(textons_filename, 'rb') as file:  
    textons = pickle.load(file)

clusters= 16*16 #optimum clusters

# Calculate textons representation and assign them
test_hist=[] 
for i in range (0,10*ind):
    tmap= assignTextons(fbRun(fb,img[i]),textons.transpose())
    hist= histc(tmap.flatten(),np.arange(clusters))/tmap.size
    test_hist.append(hist)

#Predict labels of the test set
model_filename = "model_new.pkl"  
with open(model_filename, 'rb') as file:  
    model = pickle.load(file)

prediction= model.predict(test_hist)

ACA= accuracy_score(l,prediction)
CM=confusion_matrix(l,prediction)

plt.figure()
plot_confusion_matrix.plot_confusion_matrix(CM,posible_labels,normalize=True,title='Normalize confusion matrix of test set')
plt.show()

print ('ACA measure of test set is ' + str(ACA))


#Sources
#https://scikit-learn.org/stable/modules/classes.html#module-sklearn.neighbors
