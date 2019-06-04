import os
import numpy as np
# from sklearn import preprocessing
# import csv 
import time
import random
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import pandas as pd
import h5py

from keras.models import Model, Sequential,save_model
from keras.layers.normalization import BatchNormalization
from keras.layers import (
    Dense,
    Activation,
    Dropout,
    Conv2D, 
    MaxPooling2D,
    Input,
    Flatten) 
from keras.optimizers import Adam
from keras.utils import plot_model
from keras import backend as K

from keras.preprocessing.image import ImageDataGenerator                 
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping,TensorBoard, ModelCheckpoint

import tensorflow as tf                                                                                       
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))

import resnet_AP


def mkdir(path):
    isExist = os.path.exists(path)
    if isExist:
        print("the path exists")
        return False
    else:
        os.makedirs(path)
        print("the path is created successfully")
    return True

def MPE(y_pred, y_true):
    # define the Mean Positioning Error
    d= y_pred - y_true
    loss = K.sqrt(K.batch_dot(d,d,axes =[1,1])) 
    # print(K.shape(loss),K.int_shape(loss))
    return K.mean(loss) 

def keras_debug(root_path, newpath):
    
    # The data, shuffled and split between train and test sets:
    f = h5py.File('./dataset/224_random_18000/dataset.hdf5','r')
    X_train = f['x_train'].value
    Y_train = f['y_train'].value
    X_test = f['x_test'].value
    Y_test = f['y_test'].value
    
    X_train = X_train.astype('float32')
    Y_train = Y_train.astype('float32')
    X_validation= X_test.astype('float32')[np.arange(1,len(X_test),2)]
    Y_validation= Y_test.astype('float32')[np.arange(1,len(Y_test),2)]
    X_test = X_test.astype('float32')[np.arange(0,len(X_test),2)]
    Y_test = Y_test.astype('float32')[np.arange(0,len(Y_test),2)]
    
    # subtract mean and normalize
    mean_image = np.mean(X_train, axis=0)
    
    X_train -= mean_image
    X_test -= mean_image
    X_validation -= mean_image
    X_train /= 128.
    X_test /= 128.
    X_validation /=128.

    # output consists of 3D coordinate(3), euler vector(2) and euler angle(1). 
    # The last three need to be preprocess.
    
    Y_test[:,3:4]=Y_test[:,3:4]*3000
    Y_train[:,3:4] = Y_train[:,3:4]*3000
    Y_validation[:,3:4] = Y_validation[:,3:4]*3000
    
    Y_test[:,5] = Y_test[:,5]*3000/np.pi
    Y_train[:,5] = Y_train[:,5]*3000/np.pi
    Y_validation[:,5] = Y_validation[:,5]*3000/np.pi
    print(Y_test)

    # create a CNN network

    model = resnet_AP.ResnetBuilder.build_resnet_101((img_rows, img_cols, img_channels),6)

    # compile and plot the network 
    model.compile(loss = 'mse',
            optimizer = 'adam',
            loss_weight = [1,1],
            metrics = [MPE])

    plot_model(model, to_file = newpath+'model_store/model.png', 
            show_shapes =True,
            show_layer_names = True)

    ############################### network parameters ###################################
    batch_size = 64 
    epochs = 1000
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=1000, min_lr=0.5e-6)
    early_stopper = EarlyStopping(min_delta= 0.1, patience=200)
    csv_logger = CSVLogger(newpath + 'logs.csv')

    tensorboard = TensorBoard(log_dir = newpath+'tensorboard',
            write_graph = True)
    checkpoint = ModelCheckpoint(filepath=newpath+'model_store/best_model.hdf5',
            monitor = 'val_loss',
            save_best_only = True,
            save_weights_only = False,
            mode = 'min',
            period = 1)
    callback_list = [lr_reducer, early_stopper,  csv_logger, tensorboard, checkpoint]
    #######################################################################################

    train_step = model.fit(X_train, Y_train,
            batch_size =batch_size,
            epochs = epochs,
            validation_data = (X_validation, Y_validation),
            shuffle = True,
            callbacks = callback_list)

    mpe = train_step.history['val_MPE']
    min_mpe = float('%.2f' % np.min(mpe))
    print("min_MPE is :", min_mpe)
    time_now = str(time.ctime())
    os.rename(newpath,root_path+str(min_mpe)+"_"+ str(epochs)+ "_"+time_now)
    print("training and saving is completed!")




if __name__ =='__main__':
    time_start = time.time() 

    root_path = "./debug/debug_Resnet/224_random/"
    mkdir(root_path)
    file_name = []
    for files in os.listdir(root_path):
        file_name .append(files)
    for i in range (100):
        if not(str(i) in file_name):
            proc_num = i
            break


    newpath = root_path + str(proc_num) + "/"
    mkdir(newpath)
    mkdir(newpath+'model_store/')
    keras_debug(root_path, newpath)

    time_end = time.time()
    print((time_end - time_start)/3600)
