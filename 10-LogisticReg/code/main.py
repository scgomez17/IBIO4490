#!/usr/bin/ipython3

# read kaggle facial expression recognition challenge dataset (fer2013.csv)
# https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge

#Import lybraries
import os 
import glob
import zipfile 
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import plot_confusion_matrix 
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
import timeit
import pickle
from PIL import Image
from skimage import color
from cv2 import cv2

#Download and/or decompress of the database
data_zip= os.path.isdir('fer2013.zip')
data_name= os.path.isdir('fer2013')

if (data_zip==True and data_name==False):
    data= zipfile.ZipFile('fer2013.zip','r')
    data.extractall('fer2013')
    data.close
elif (data_zip==False and data_name==False):
    url_data= 'https://www.dropbox.com/s/i53z586d5ygimzb/fer2013..zip?dl=1'
    urllib.request.urlretrieve(url_data,'fer2013.zip')
    data= zipfile.ZipFile('fer2013.zip','r')
    data.extractall('fer2013')
    data.close

def sigmoid(x):
    return 1/(1+np.exp(-x))

def get_data():
    # angry, disgust, fear, happy, sad, surprise, neutral
    with open("fer2013/fer2013.csv") as f:
        content = f.readlines()

    lines = np.array(content)
    num_of_instances = lines.size
    print("number of instances: ",num_of_instances)
    print("instance length: ",len(lines[1].split(",")[1].split(" ")))

    x_train, y_train, x_test, y_test = [], [], [], []

    for i in range(1,num_of_instances):
        emotion, img, usage = lines[i].split(",")
        pixels = np.array(img.split(" "), 'float32')
        emotion = 1 if int(emotion)==3 else 0 # Only for happiness
        if 'Training' in usage:
            y_train.append(emotion)
            x_train.append(pixels)
        elif 'PublicTest' in usage:
            y_test.append(emotion)
            x_test.append(pixels)

    x_val= x_train.copy
    x_val= x_train[20000:len(x_train)]
    y_val= y_train.copy
    y_val= y_train[20000:len(y_train)]
    x_train= x_train[0:20000]
    y_train= y_train[0:20000]

    #------------------------------
    #data transformation for train and test sets
    x_train = np.array(x_train, 'float64')
    y_train = np.array(y_train, 'float64')
    x_val = np.array(x_val, 'float64')
    y_val = np.array(y_val, 'float64')
    x_test = np.array(x_test, 'float64')
    y_test = np.array(y_test, 'float64')

    x_train /= 255 #normalize inputs between [0, 1]
    x_val /= 255
    x_test /= 255

    x_train = x_train.reshape(x_train.shape[0], 48, 48)
    x_val= x_val.reshape(x_val.shape[0],48,48)
    x_test = x_test.reshape(x_test.shape[0], 48, 48)
    y_train = y_train.reshape(y_train.shape[0], 1)
    y_val = y_val.reshape(y_val.shape[0], 1)
    y_test = y_test.reshape(y_test.shape[0], 1)

    print(x_train.shape[0], 'train samples')
    print(x_val.shape[0], 'val samples')
    print(x_test.shape[0], 'test samples')

    return x_train, y_train, x_val, y_val, x_test, y_test

class Model():
    def __init__(self):
        if args.test == True or args.demo == True:
            w_binario = "w_binario_.pkl"  
            b_binario = "b_binario_.pkl"    
            with open(w_binario, 'rb') as file:  
                w = pickle.load(file)
            with open(b_binario, 'rb') as file:  
                b = pickle.load(file)
            self.W= w
            self.b=b
        else:    
            params = 48*48 # image reshape
            out = 1 # smile label
            self.lr = 0.0001 # Change if you want
            self.W = np.random.randn(params, out) #random inicialization
            self.b = np.random.randn(out) #random inicialization

    def forward(self, image):
        image = image.reshape(image.shape[0], -1)
        out = np.dot(image, self.W) + self.b
        return out

    def compute_loss(self, pred, gt):
        J = (-1/pred.shape[0]) * np.sum(np.multiply(gt, np.log(sigmoid(pred))) + np.multiply((1-gt), np.log(1 - sigmoid(pred)))) #Error function
        return J

    def compute_gradient(self, image, pred, gt):
        image = image.reshape(image.shape[0], -1)
        W_grad = np.dot(image.T, pred-gt)/image.shape[0]
        self.W -= W_grad*self.lr

        b_grad = np.sum(pred-gt)/image.shape[0]
        self.b -= b_grad*self.lr

def train(model):
    x_train, y_train, x_val, y_val, _ , _  = get_data()
    batch_size = 50 # Change if you want
    epochs = 40000 # Change if you want
    Loss_train=[]
    Loss_x2=[] #val or test
    for i in range(epochs):
        loss = []
        for j in range(0,x_train.shape[0], batch_size):
            _x_train = x_train[j:j+batch_size]
            _y_train = y_train[j:j+batch_size]
            out = model.forward(_x_train) #predictions
            loss.append(model.compute_loss(out, _y_train))
            model.compute_gradient(_x_train, out, _y_train)
        out = model.forward(x_val) #Predictions 
        loss_val = model.compute_loss(out, y_val) 
        print('Epoch {:6d}: train: {:.5f} | val: {:.5f}'.format(i, np.array(loss).mean(), loss_val))
        Loss_train.append(np.array(loss).mean())
        Loss_x2.append(loss_val)
        if i==epochs-1:
            plot(range(epochs), Loss_train, Loss_x2 , 'Val','optimal')
            w_binario = "w_binario_.pkl"  
            b_binario = "b_binario_.pkl"  
            with open(w_binario, 'wb') as file:  
                pickle.dump(model.W, file)
            with open(b_binario, 'wb') as file:  
                pickle.dump(model.b, file)
     
def plot(vector_x,train,vector_y2,plot_label,name): # Add arguments
    #vector_x2 --> val/test
    plt.plot(vector_x,train,label="train")   
    plt.plot(vector_x,vector_y2,label=plot_label)   
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Prediction error')
    plt.title('Prediction error vs epochs')
    plt.savefig(name+ '.png')
    plt.show()
    
    pass

def test(model):
    _, _, _, _, x_test, y_test = get_data()
   
    out = model.forward(x_test) #Predictions 
    out= sigmoid(out)
    
    #PR curve
    p=[]
    r=[]
    valmin=1
    auxF=0
    count=0
    count2=0
    for th in np.arange(0,1,0.1):
        TP=0
        FP=0
        FN=0
        for j in range(0,len(out)):
                if out[j][0]>=th and y_test[j][0]==1:
                    TP+=1
                elif out[j][0]<th and y_test[j][0]==1:
                    FN+=1
                elif out[j][0]>=th and y_test[j][0]==0:
                    FP+=1
        p.append(TP/(TP+FP))
        r.append(TP/(TP+FN))
        valcmp=np.sqrt((1-p[count])**2+(1-r[count])**2)
        if valmin>valcmp:
            valmin=valcmp
            auxF=th
            count2=count
        count+=1
    F1=2*((p[count2]*r[count2])/(p[count2]+r[count2]))
    print('The F1 measure is '+str(F1))

    r.sort()
    p.sort(reverse=True)
    plt.plot(r,p)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR curve for binary classification. F1 measure is:  ' + str(F1))
    plt.savefig('Pr_binario.png')
    plt.show()
    print(out[10])
    
    #Confusion matrix and ACA performance
    ACA=np.zeros((2,2))
    for k in range(0,len(out)):
        if out[k]>=0.5 and y_test[k] ==1:
            ACA[1][1]=ACA[1][1]+1
        elif out[k]>=0.5 and y_test[k]==0:
            ACA[1][0]=ACA[1][0]+1
        elif out[k]<0.5 and y_test[k]==1:
            ACA[0][1]=ACA[0][1]+1
        elif out[k]<0.5 and y_test[k]==0:
            ACA[0][0]=ACA[0][0]+1

    plt.figure()
    plot_confusion_matrix.plot_confusion_matrix(ACA,range(0,2),normalize=True,title='Normalize confusion matrix of test set')
    plt.savefig('CM_binario.png')
    plt.show()
    ACA= np.mean(np.diag(ACA))
    print ('ACA is' + str(ACA))
    pass


def listaAleatorios(n):
    import random
    lista = [0]  * n
    for i in range(n):
        lista[i] = random.randint(0, 21)
    return lista


def demo(model,w):
    
    if w== 'natural':

        img_path= os.path.join('../Natural_Images','*.jpg')
        img_path=glob.glob(img_path)
        images= []
        for number in range(len(img_path)):
            images.append(cv2.imread(os.path.join(img_path[number])))
            images[number]=cv2.resize(images[number],(48,48))
            images[number] = color.rgb2gray(images[number])
        
        images = np.array(images)
        images = images.astype('float64')

        out = model.forward(images) #Predictions 
        out= sigmoid(out)
        out[out>= 0.5] =1
        out[out< 0.5] = 0

        count=0
        for j in listaAleatorios(8):
            plt.figure(1)
            plt.subplot(2,4,count+1)
            plt.imshow(images[j],'gray')
            plt.axis('off')
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            plt.text(1,1,str(out[j]),fontsize=20,bbox=props)
            if count == 7:
                plt.show()
            count=count+1
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', default= False)
    parser.add_argument('--demo', action='store_true', default= False)
    args = parser.parse_args()

    if args.test == True:
        model = Model()
        test(model)    
    elif args.demo == True:
        model = Model()
        #natural for show demo in natural images
        #test for show demo with test images
        demo(model,'natural')
    else:     
        tic= timeit.default_timer() 
        for em in range(8):
            model = Model()
            train(model)
        toc= timeit.default_timer()       
        train_time= toc-tic
        print ('The train time process is:' + str (train_time))
        args.test=True
        model = Model()
        test(model)

