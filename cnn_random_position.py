import os
import numpy as np
# from sklearn import preprocessing
import csv 
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
    print(K.shape(loss),K.int_shape(loss))
    return K.mean(loss) 

def _bn_relu(input):
    """To build a BN --> relu block
    """
    norm = BatchNormalization(axis = CHANNEL_AXIS)(input)
    return Activation("relu")(norm)

def _conv_bn_relu(**conv_params):
    """To build a conv ->BN -> relu block
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                strides=strides, padding=padding,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer)(input)
        return _bn_relu(conv)


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

    # create a CNN network
    output_shape = 6
    input_shape = mean_image.shape
    input = Input(shape = input_shape)
    # bconv0 = Conv2D(filters = 3, kernel_size = (7,7),
            # strides = (4,4),
            # activation = 'relu',
            # padding = 'same')(input)
    # bpool0 = MaxPooling2D(pool_size =(3,3), strides=(2,2),
            # padding = 'same')(bconv0)
    conv1 = Conv2D(filters = 32, kernel_size=(3,3),
            strides = (1,1),
            activation = 'relu',
            padding = 'same')(input)
    pool2 = MaxPooling2D(pool_size = (3,3),strides=(2,2),
            padding = 'same')(conv1)
    conv2 = Conv2D(filters = 64, kernel_size = (3,3),
            activation = 'relu',
            padding = 'same')(pool2)
    pool3 = MaxPooling2D(pool_size = (3,3), strides = (2,2),
            padding = 'same')(conv2)
    conv3 = Conv2D(filters = 128, kernel_size = (3,3),
            activation = 'relu',
            padding = 'same')(pool3)
    pool4 = MaxPooling2D(pool_size = (3,3), strides =(2,2))(conv3)
    conv4 = Conv2D(filters = 256, kernel_size = (3,3),
            activation = 'relu')(pool4)
    # pool4 = MaxPooling2D(pool_size = (3,3),strides=(3,3))(conv3)

    flatten1 = Flatten()(conv4)
    dense1 = Dense(1024, activation = 'relu')(flatten1)
    dense2 = Dense(512, activation = 'relu')(dense1)
    dense3 = Dense(256, activation = 'relu')(dense2)
    
    dense = Dense(output_shape)(dense3)

    # compile and plot the network 
    model = Model(inputs = input, outputs = dense)
    model.compile(loss = 'mse',
            optimizer = 'adam',
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
    checkpoint = ModelCheckpoint(filepath=newpath+'model_store/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
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
    os.rename(newpath,root_path +str(min_mpe)+"_"+ str(epochs)+ "_"+time_now)
    print("training and saving is completed!")




    ####################### plan B for output ########################################33
    # x_axis = []
    # train_loss = []
    # test_loss = []
    # validation_loss = []

    # batch_size = 32
    # epochs = 1 
    # for i in range(10000):
        # x_axis.append(i)
        # print("*"*50, "\n step: ",i)
        # train_step = model.fit (X_train, Y_train,
                # batch_size = batch_size,
                # epochs = epochs,
                # validation_data = (X_test, Y_test),
                # shuffle = True,
                # callbacks = [lr_reducer, early_stopper, csv_logger])

        # train_loss.append(train_step.history['loss'][0])
        # validation_loss.append(train_step.history['val_loss'][0])
        
        # output_model = Model(inputs = input, outputs =model.get_layer('dense_3').output)
        # output_final = output_model.predict(X_train)
        # print(output_final)
    ##################################################################################


if __name__ =='__main__':
    time_start = time.time() 

    root_path = "./debug/debug_CNN/224_random/"
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
