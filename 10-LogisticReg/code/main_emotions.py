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

def get_data(em):
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
        if em <=6:
            emotion = 1 if int(emotion)==em else 0 # Only for happiness
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
            w0 = "w_multiclase_0.pkl"  
            w1 = "w_multiclase_1.pkl"  
            w2 = "w_multiclase_2.pkl"  
            w3 = "w_multiclase_3.pkl"  
            w4 = "w_multiclase_4.pkl"  
            w5 = "w_multiclase_5.pkl"  
            w6 = "w_multiclase_6.pkl"  

            b0 = "b_multiclase_0.pkl"  
            b1 = "b_multiclase_1.pkl"  
            b2 = "b_multiclase_2.pkl"  
            b3 = "b_multiclase_3.pkl"  
            b4 = "b_multiclase_4.pkl"  
            b5 = "b_multiclase_5.pkl"  
            b6 = "b_multiclase_6.pkl"    
            
            with open(w0, 'rb') as file:  
                w0 = pickle.load(file)
            with open(w1, 'rb') as file:  
                w1 = pickle.load(file)
            with open(w2, 'rb') as file:  
                w2 = pickle.load(file)
            with open(w3, 'rb') as file:  
                w3 = pickle.load(file)
            with open(w4, 'rb') as file:  
                w4 = pickle.load(file)
            with open(w5, 'rb') as file:  
                w5 = pickle.load(file)
            with open(w6, 'rb') as file:  
                w6 = pickle.load(file)

            with open(b0, 'rb') as file:  
                b0 = pickle.load(file)
            with open(b1, 'rb') as file:  
                b1 = pickle.load(file)
            with open(b2, 'rb') as file:  
                b2 = pickle.load(file)
            with open(b3, 'rb') as file:  
                b3 = pickle.load(file)
            with open(b4, 'rb') as file:  
                b4 = pickle.load(file)
            with open(b5, 'rb') as file:  
                b5 = pickle.load(file)
            with open(b6, 'rb') as file:  
                b6 = pickle.load(file)
        
            self.W0= w0 
            self.W1= w1
            self.W2= w2
            self.W3= w3
            self.W4= w4
            self.W5= w5
            self.W6= w6
            
            self.b0= b0 
            self.b1= b1
            self.b2= b2
            self.b3= b3
            self.b4= b4
            self.b5= b5
            self.b6= b6
        else:    
            params = 48*48 # image reshape
            out = 1 # smile label
            self.lr = 0.0001 # Change if you want
            self.W = np.random.randn(params, out) #random inicialization
            self.b = np.random.randn(out) #random inicialization

    def forward(self, image,em):
        image = image.reshape(image.shape[0], -1)
        if em ==0:
            out = np.dot(image, self.W0) + self.b0
        elif em ==1:
            out = np.dot(image, self.W1) + self.b1
        elif em ==2:
            out = np.dot(image, self.W2) + self.b2
        elif em ==3:
            out = np.dot(image, self.W3) + self.b3
        elif em ==4:
            out = np.dot(image, self.W4) + self.b4
        elif em ==5:
            out = np.dot(image, self.W5) + self.b5
        elif em ==6:
            out = np.dot(image, self.W6) + self.b6
        else:
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
        
def train(model,em):
    x_train, y_train, x_val, y_val, _ , _  = get_data(em)
    batch_size = 50 # Change if you want
    epochs = 40000 # Change if you want
    Loss_train=[]
    Loss_x2=[] #val or test
    for i in range(epochs):
        loss = []
        for j in range(0,x_train.shape[0], batch_size):
            _x_train = x_train[j:j+batch_size]
            _y_train = y_train[j:j+batch_size]
            out = model.forward(_x_train,7) #predictions
            loss.append(model.compute_loss(out, _y_train))
            model.compute_gradient(_x_train, out, _y_train)
        out = model.forward(x_val,7) #Predictions 
        loss_val = model.compute_loss(out, y_val) 
        print('Epoch {:6d}: train: {:.5f} | val: {:.5f}'.format(i, np.array(loss).mean(), loss_val))   
        Loss_train.append(np.array(loss).mean())
        Loss_x2.append(loss_val)
        if i==epochs-1:
            w_multiclase = 'w_multiclase_'+str(em)+'.pkl'  
            b_multiclase = 'b_multiclase_'+str(em)+'.pkl'  
            with open(w_multiclase, 'wb') as file:  
                pickle.dump(model.W, file)
            with open(b_multiclase, 'wb') as file:  
                pickle.dump(model.b, file)
            plot(range(epochs), Loss_train, Loss_x2 , 'Val','multiclase '+str(em))

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

    prob= np.zeros([3589,7])
    for em in range(7):
        _, _, _, _, x_test, y_test = get_data(em)
        out = model.forward(x_test,em) #Predictions 
        out= sigmoid(out)
        for rows in range(len(out)):
            prob [rows,em]= out [rows]

    for i in range (3589):
        label= np.where (prob[i,:] == max(prob[i,:]))
        out[i]= label[0]

    _, _, _, _, _, y_test = get_data(7)

    #Confusion matrix and ACA performance
    CM=confusion_matrix(y_test,out)
    plt.figure()
    plot_confusion_matrix.plot_confusion_matrix(CM,range(7),normalize=True,title='Normalize confusion matrix')
    plt.savefig('CM_Multiclase.png')
    plt.show()
    ACA= accuracy_score(y_test,out)
    print('The ACA is {:.5f}'.format(ACA))

    pass

def listaAleatorios(n):
    import random
    lista = [0]  * n
    for i in range(n):
        lista[i] = random.randint(0, 21)
    return lista


def demo(model,w):
    
    if w== 'natural':
        prob= np.zeros([22,7])
        img_path= os.path.join('../Natural_Images','*.jpg')
        img_path=glob.glob(img_path)
        images= []
        for number in range(len(img_path)):
            images.append(cv2.imread(os.path.join(img_path[number])))
            images[number]=cv2.resize(images[number],(48,48))
            images[number] = color.rgb2gray(images[number])
        
        images = np.array(images)
        images = images.astype('float64')

        for em in range(7):
            out = model.forward(images,em) #Predictions 
            out= sigmoid(out)
            for rows in range(len(out)):
                prob [rows,em]= out [rows]
        for i in range (22):
            label= np.where (prob[i,:] == max(prob[i,:]))
            out[i]= label[0]
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
            train(model,em)
        toc= timeit.default_timer()       
        train_time= toc-tic
        print ('The train time process is:' + str (train_time))
        args.test=True
        model = Model()
        test(model)



