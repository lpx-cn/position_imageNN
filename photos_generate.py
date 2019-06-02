import os
import time
import numpy as np
import math 
from math import pi
import time
import random
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
import cv2
import h5py

def mkdir(path):
 
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path) 
 
        print(path+' is created.') 
        return True
    else:
        print(path+' is existed.') 
        return False
    
class room(object):
    def __init__(self, room_size = [6000,6000,5000],
            LED_ordinate = np.mat([(2500,3000,5000),(3500,3000,5000),(3000,3500,5000)]).T,
            lm = 250):

        self.room_size = room_size
        self.LED_ordinate = LED_ordinate
        self.lm = lm

    def move_led(self, led):
        self.LED_ordinate = led

    def change_room(self,room_size):
        self.room_size= room_size
    def change_lm(self,lm):
        self.lm = lm

class camera(room):

    def __init__(self, 
            room_size = [6000, 6000, 5000],
            LED_ordinate = np.mat([(2500,3000,5000),(3500,3000,5000),(3000,3500,5000)]).T,
            lm = 250,
            position=np.mat([[3000],[3000],[1500]]),
            euler=[1,0,0], #alpha, belta, phi
            focal = 15,
            sensor = [36, 24],
            resolution = [28,28]):
        room.__init__(self,room_size,LED_ordinate,lm)

        self.position = position
        self.euler = euler
        self.focal = focal
        self.sensor_size = sensor
        self.resolution=resolution
        self.interval=[sensor[0]/resolution[0],sensor[1]/resolution[1]]
        self.photo_size=resolution
        self.__euler2direction()
        self.isThree = None
    
    def move(self,position):
        self.position=position
        
    def rorate(self, euler):
        self.euler = euler
        self.__euler2direction()
        
    def __euler2direction(self):
        # euler method and direcion vector
        p1 = self.euler[0]
        p2 = self.euler[1]
        p3 = math.sqrt(1-p1**2-p2**2)
        phi = self.euler[2]
        t=(1-math.cos(phi))
        self.direction = np.array([(p1**2*t+math.cos(phi), p2*p1*t-p3*math.sin(phi), p3*p1*t+p2*math.sin(phi)), 
            (p2*p1*t+p3*math.sin(phi), p2**2*t+math.cos(phi), p3*p2*t-p1*math.sin(phi)),
            (p3*p1*t-p2*math.sin(phi), p3*p2*t+p1*math.sin(phi), p3**2*t+math.cos(phi))])
        self.x_dir = self.direction[:,0]
        self.y_dir = self.direction[:,1]
        self.z_dir = self.direction[:,2]

    def take_photo(self):
        img = np.zeros((self.resolution[0],self.resolution[1],3),np.uint8)
        size = np.shape(img)

        R=0
        G=0
        B=0
        
        for i in range (size[0]):
            for j in range (size[1]):
                img[i,j,:] = self.__PixelValue(i,j) 
                if img[i,j,0]==255:
                    R = 1
                elif img[i,j,1]==255:
                    G=1
                elif img[i,j,2]==255:
                    B=1

        if R==1 and G==1 and B==1:
            self.isThree=True
        else:
            self.isThree=False
        

        # cv2.namedWindow("img")
        # cv2.imshow("img",img)
        # cv2.waitKey(5000)
     
        return img

    def __PixelValue(self,i,j):
        pixel_ccs = [i-self.resolution[0]/2,j-self.resolution[1]/2]
        pixel_ccs = np.multiply(pixel_ccs ,self.interval)
        pixel_ccs = np.hstack((pixel_ccs, -self.focal))
        pixel_eccs = np.mat(np.hstack((pixel_ccs,1)))
        pixel_eccs = pixel_eccs.T

        trans_matrix = np.hstack((self.direction, self.position))
        trans_matrix = np.vstack((trans_matrix,np.array([0,0,0,1])))
        trans_matrix = np.mat(trans_matrix)
        # print("trans_matrix:",trans_matrix)

        pixel_ewcs = trans_matrix * pixel_eccs        
        pixel_wcs = np.mat(pixel_ewcs[0:3])
        # print((pixel_ewcs))

        pixel_value = []
        for i in range(3):
            RGBi = self.isInside(pixel_wcs, led=self.LED_ordinate[:,i],r=self.lm,c=self.position)
            pixel_value.append(RGBi)
        # print(pixel_value)

        # white process
        # if np.linalg.norm(pixel_value) >1:
            # pixel_value = [255,255,255]

        return pixel_value

    def isInside(self, wcs, led, r, c):
        # Caculate the value of the function with pixel's wcs coordinate
        wcs = np.squeeze(np.asarray(wcs))
        led = np.squeeze(np.asarray(led))
        r = np.squeeze(np.asarray(r))
        c= np.squeeze(np.asarray(c))

        t = 1 + np.dot((led - wcs),(c-led))/np.linalg.norm(c-led,2)**2
        results = np.linalg.norm(1/t*(wcs - c)+c-led)**2 - r**2
        # print("result:",results)

        if results <= 0:
            return 255
        else:
            return 0


def data_generate_uniformity(c1, x,y,z,path):
    # Obtain the data which is a uniform distribution.
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    x_blank = []
    y_blank = []
    
    x_axis = np.linspace(0, 6000, x)
    y_axis = np.linspace(0, 6000, y)
    z_axis = np.linspace(0, 4000, z) 

    xg = 6000/(x-1)/2
    yg = 6000/(y-1)/2
    zg = 4000/(z-1)/2

    mkdir(path + "train_photo/")
    mkdir(path + "test_photo/")
    mkdir(path + "blank_photo/")
    
    for i in x_axis:
        for j in y_axis:
            for k in z_axis:
                time_start = time.time()
    
                c1.move(np.mat([i,j,k]).T)
                img = c1.take_photo()
                if c1.isThree: # Only use the photos with 3 LEDs.
                    cv2.imwrite(path+"train_photo/"+str(i)+"_"+str(j)+"_"+str(k)+".jpg", img)
                    x_train.append(img)
                    y_train.append(([i,j,k]))
                else:
                    cv2.imwrite(path+"blank_photo/"+str(i)+"_"+str(j)+"_"+str(k)+".jpg", img)
                    x_blank.append(img)
                    y_blank.append(([i,j,k]))



                i1 = i+xg
                j1 = j+yg
                k1 = k+zg
                if i1<=6000 and j1<=6000 and k1<=4000:
                    c1.move(np.mat([i1,j1,k1]).T)
                    img = c1.take_photo()
                    if c1.isThree:
                        cv2.imwrite(path+"test_photo/"+str(i1)+"_"+str(j1)+"_"+str(k1)+".jpg", img)
                        x_test.append(img)
                        y_test.append(([i1,j1,k1]))
                    else:
                        cv2.imwrite(path+"blank_photo/"+str(i)+"_"+str(j)+"_"+str(k)+".jpg", img)
                        x_blank.append(img)
                        y_blank.append(([i,j,k]))
                        

                
                time_end = time.time()
                print("time:",time_end-time_start)
                print(i,j,k,c1.isThree)
    
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    f = h5py.File(path+'dataset.hdf5', 'a')
    f.create_dataset('x_train', data = x_train)
    f.create_dataset('y_train', data = y_train)
    f.create_dataset('x_test', data = x_test)
    f.create_dataset('y_test', data = y_test)
    f.create_dataset('y_blank',data = y_blank)

    return f

def data_generate_random(c1, num, rate, path):
    # obtain the data in random position
    #   c1: the camera() class
    #   num: the number for all instance
    #   rate: the number of test data / num of all
    #   path: data file's path

    def dataset_add_data(dataset,data):
        shape = list(dataset.shape)
        shape[0]+=1
        dataset.resize(tuple(shape))
        dataset[shape[0]-1] = data

    mkdir(path + "train_photo/")
    mkdir(path + "test_photo/")

    f = h5py.File(path+'dataset.hdf5', 'a')
    xshape = [0]+c1.resolution +[3]
    max_xshape = [None] + xshape[1:4] 
    yshape = [0,6]
    max_yshape = [None,6]
    f.create_dataset('x_train', xshape, maxshape=max_xshape, dtype='float32')
    f.create_dataset('y_train', yshape, maxshape=max_yshape, dtype='float32')
    f.create_dataset('x_test', xshape, maxshape=max_xshape, dtype='float32')
    f.create_dataset('y_test', yshape, maxshape=max_yshape, dtype='float32')
    

    num_right=0
    num_wrong=0
    x_test=f['x_test']
    y_test=f['y_test']
    x_train=f['x_train']
    y_train=f['y_train']

    i=0
    count = 0
    while 1:
        t_strat = time.time()
        if i>=num:
            break

        y = []
        for size in c1.room_size:
            y.append(random.uniform(0,size))
        Y=np.mat(y).T

        new_euler = []
        new_euler.append(random.uniform(-1,1))
        beta=np.sqrt(1-new_euler[0]**2)
        new_euler.append(random.uniform(-beta,beta))
        new_euler.append(random.uniform(-np.pi, np.pi))


        c1.move(Y)
        c1.rorate(new_euler)

        if c1.z_dir[2]>0 :
            img = c1.take_photo()
            if c1.isThree :
                if random.random() > rate:
                    cv2.imwrite(path+"train_photo/"+str(y)+str(new_euler)+".jpg", img)

                    dataset_add_data(x_train, img)
                    dataset_add_data(y_train, (y + new_euler))

                    num_right+=1
                    i+=1
                else:
                    cv2.imwrite(path+"test_photo/"+str(y)+str(new_euler)+".jpg", img)

                    dataset_add_data(x_test, img)
                    dataset_add_data(y_test, (y + new_euler))

                    num_right+=1
                    i+=1
            else:
                num_wrong+=1
        else:
            num_wrong+=1
        t_end = time.time()
        count+=1

        print("="*50)
        print("Step %d: %.2fs " %(count, t_end-t_strat))
        print("Done:%d/%d, trainset:%d, testset:%d" %(i,num,y_train.shape[0],y_test.shape[0]))


    print("completed! wrong/right:", num_wrong/num_right)


    return f


if __name__ =='__main__':
    path = "./dataset_28_28/"
    if mkdir(path):
        c1 = camera()
        data_generate_uniformity(c1, 30, 30, 30, path)
    
        # data_generate_random(c1, 40000, 0.4, path)


# time.sleep(2)
