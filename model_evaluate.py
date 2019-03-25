import os
import numpy as np
from sklearn import preprocessing
import csv 
import time
import random
import matplotlib
matplotlib.use('Agg')
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

def MPE(y_pred, y_true):
    # define the Mean Positioning Error
    d= y_pred - y_true
    loss = K.sqrt(K.batch_dot(d,d,axes =[1,1])) 
    # print(K.shape(loss),K.int_shape(loss))
    return K.mean(loss) 

def mkdir(path):
    isExist = os.path.exists(path)
    if isExist:
        print("the path exists")
        return False
    else:
        os.makedirs(path)
        print("the path is created successfully")
    return True

def read_data(path):
    # The data, shuffled and split between train and test sets:
    f = h5py.File(path,'r')

    X_train = f['x_train'].value
    X_test = f['x_test'].value
    Y_test = f['y_test'].value

    X_train = X_train.astype('float32')
    X_validation= X_test.astype('float32')[np.arange(1,len(X_test),2)]
    Y_validation= Y_test.astype('float32')[np.arange(1,len(Y_test),2)]
    X_test = X_test.astype('float32')[np.arange(0,len(X_test),2)]
    Y_test = Y_test.astype('float32')[np.arange(0,len(Y_test),2)]

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
        
def model_predict(path, data):
    # the path is model's path
    # the data is [X_test, Y_test, X_validation, Y_validation]


    test_features = data[0]
    test_labels = data[1]
    validation_features = data[2]
    validation_labels = data[3]

    model = load_model(path, custom_objects = {'MPE':MPE})
    test_loss = model.evaluate(test_features,test_labels,batch_size = test_labels.shape[0],verbose =0)
    val_loss = model.evaluate(validation_features,validation_labels,
                              batch_size = validation_features.shape[0],verbose = 0)
    
    # Show the prediction results 

    # loss=[]
    # loss_square = []
    # prediction = []
    
    
    # data_value = test_features
    # label_value = test_labels
    # time_list = []
    # for i in range(len(data_value)):
        # time_start = time.time()
        # x = np.expand_dims(data_value[i], axis=0)
        # predictation_i = model.predict(x)
        # prediction.append(predictation_i)
        # time_end = time.time()
        # time_list.append(time_end - time_start)
    
        # loss_i =  label_value[i] - predictation_i
        # loss_i = np.linalg.norm(loss_i)
    
        # loss_square.append(loss_i**2)
        # loss.append(loss_i)
    
        # print("="*50,'\n step: %d'% i)
        # print("the real value is: ", label_value[i])
        # print("prediction value is : ", predictation_i)
        # print("posisitoning error is : %f"%loss_i)
        
    # print(loss)
    print('test_loss and MPE:',test_loss)
    print('validation_loss and MPE:',val_loss)
    print("="*25,"completed!","="*25)
    
    # print("Mean positioning time cost:", np.mean(time_list))

    return [test_loss[1], val_loss[1]]

if __name__ == '__main__':


    model_path_list = get_model_files('.')
    data_path = './dataset_28_28/dataset.hdf5'
    mpe_list = []
    for path in model_path_list:
        data = read_data(data_path)
        mpe, mpe_val= model_predict(path,data)
        mpe_list.append(mpe)
    mpe_min = min(mpe_list)
    mpe_min_arg = mpe_list.index(mpe_min)
    mpe_min_path = model_path_list[mpe_min_arg]
    print("The min MPE is %f, in %s" %(mpe_min, mpe_min_path))
