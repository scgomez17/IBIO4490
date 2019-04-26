def obtain_labels(path):
    
    import numpy as np
    
    labels_file= path + 'list_attr_celeba.txt'

    with open(labels_file, 'rb') as File:  
        img_labels= File.readlines()
    File.close()
    img_labels= img_labels[1:len(img_labels)]
    img_labels= [str(img_labels[i]).split("\\n") for i in range(len(img_labels))]
    img_labels= [img_labels[i][0].split(',') for i in range(len(img_labels))]

    labels= []
    labels_aux= []
    for i in range(len(img_labels)):
        labels_aux.append(float(img_labels[i][16]))
        labels_aux.append(float(img_labels[i][6]))
        labels_aux.append(float(img_labels[i][9]))
        labels_aux.append(float(img_labels[i][10]))
        labels_aux.append(float(img_labels[i][12]))
        labels_aux.append(float(img_labels[i][18]))
        labels_aux.append(float(img_labels[i][21]))
        labels_aux.append(float(img_labels[i][27]))
        labels_aux.append(float(img_labels[i][32]))
        labels_aux.append(float((img_labels[i][40].split('\\r'))[0]))
        
        for i in range(len(labels_aux)):
            if labels_aux[i]==-1:
                labels_aux[i]=0
        
        labels_aux= np.array(labels_aux, 'float64')
        labels.append(labels_aux)
        labels_aux=[]

    return  labels
