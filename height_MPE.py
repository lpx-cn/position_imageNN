# the relation between height and positioning error

import os
import numpy as np
from sklearn import preprocessing
import csv 
import random
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from keras import Input,Model
from keras.models import Sequential,save_model
from keras.layers import Dense,Activation,Dropout
from keras.optimizers import Adam,TFOptimizer
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras import backend as K

from keras.models import load_model
import h5py

def mkdir(newpath):
    isExist = os.path.exists(newpath)
    if isExist:
        print("the path exists!")
        return False
    else:
        print("the path is created successfully!")
        os.makedirs(newpath)
        return True

def read_csv(filename):
    data = []
    f = open(filename,'r')
    line_reader=csv.reader(f)

    for row in line_reader:
        data.append(row)
    data = np.array(data, dtype = float)

    f.close
    return data

def read_data(path):
    # The data, shuffled and split between train and test sets:
    f = h5py.File(path,'r')

    X_train = f['x_train'].value
    X_test = f['x_test'].value
    Y_test = f['y_test'].value

    X_train = X_train.astype('float32')
    X_validation= X_test.astype('float32')[np.arange(1,len(X_test),2)]
    Y_validation= Y_test.astype('float32')[np.arange(1,len(Y_test),2)]
    X_test = X_test.astype('float32')#[np.arange(0,len(X_test),2)]
    Y_test = Y_test.astype('float32')#[np.arange(0,len(Y_test),2)]

    # subtract mean and normalize
    mean_image = np.mean(X_train, axis=0)

    X_test -= mean_image
    X_validation -= mean_image
    X_test /= 128.
    X_validation /=128.

    return [X_test, Y_test, X_validation, Y_validation]

def get_model_files(root):
    model_path_list = []
    for fpaths, dirs,fs in os.walk(root):
        for files in fs:
            if files=='best_model.hdf5':
                model_path = os.path.join(fpaths,files)
                model_path_list.append(model_path)
    return model_path_list

def MPE(y_pred, y_true):
    # define the Mean Positioning Error
    d= y_pred - y_true
    loss = K.sqrt(K.batch_dot(d,d,axes =[1,1])) 
    # print(K.shape(loss),K.int_shape(loss))
    return K.mean(loss) 

def calculate_height_MPE(path, data):
    # calculate the MPE-height of the model in the 'path' using the 'data'

    test_features = data[0]
    test_labels = data[1]
    validation_features = data[2]
    validation_labels = data[3]

    newpath = "./height_MPE/"
    mkdir(newpath)


    model = load_model(path, custom_objects = {'MPE':MPE})
    test_loss = model.evaluate(test_features,test_labels,batch_size = test_labels.shape[0],verbose =0)
    val_loss = model.evaluate(validation_features,validation_labels,
                              batch_size = validation_features.shape[0],verbose = 0)

    loss=[]
    prediction = []
    
    data_value = test_features
    label_value = test_labels
    height = []
    loss_height_dict = {}
    for i in range(len(data_value)):
        x = np.expand_dims(data_value[i], axis=0)
        predictation_i = model.predict(x)
        prediction.append(predictation_i)
    
        loss_i =  label_value[i] - predictation_i
        loss_i = np.linalg.norm(loss_i)
    
        if not(label_value[i][2] in height):
            height.append(label_value[i][2])
            loss_height_dict[label_value[i][2]]=[]

        loss_height_dict[label_value[i][2]].append(loss_i)
        loss.append(loss_i)

        # print("*"*50,'\n step: %d'% i)
        # print("the real value is: ", label_value[i])
        # print("prediction value is : ", predictation_i)
        # print("posisitoning error is : %f"%loss_i)
        
    # print("the mean loss is :%f" % (np.mean(loss)))
    # dataframe = pd.DataFrame({"test_loss":loss})
    # dataframe.to_csv(newpath + 'MPE.csv')
    
    
    meanloss_height = []
    height_MPE = {}
    for h in height:
        loss_height = np.mean(loss_height_dict[h])
        meanloss_height.append(loss_height)
        height_MPE[h]=loss_height

    height_MPE = sorted(height_MPE.items(), key=lambda x:x[0])
    for i in height_MPE:
        print(i, len(loss_height_dict[i[0]]))

    height_MPE = np.array(height_MPE)

    p = plt.figure()
    plt.plot(height_MPE[:,0],height_MPE[:,1],'-o')
    # plt.plot(height_MPE[:,0],height_MPE[:,1])
    # plt.plot([1,2,3],[4,5,6])
    plt.show()

    dataframe = pd.DataFrame({"height":height,"test_loss_mean":meanloss_height})
    dataframe.to_csv(newpath + 'MPE_height.csv')
    
if __name__ == '__main__':
    data_path = './dataset_28_28/dataset.hdf5'
    data = read_data(data_path)

    model_path_list =get_model_files('.') 
    
    for mp in model_path_list:
        calculate_height_MPE(mp, data)

