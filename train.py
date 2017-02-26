#!/usr/bin/python3
# Load the modules
import pickle
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
# Import keras deep learning libraries
import json
from sklearn.model_selection import train_test_split
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import TensorBoard
from keras.utils import np_utils
import glob
import cv2
import json
import os
import h5py

from detect import process_image

#set random seed for reproducibility
random_seed=123456
np.random.seed(random_seed)

cars_files=glob.glob('datasets/vehicles/*/*.png')
noncars_files=glob.glob('datasets/non-vehicles/*/*.png')

print('Cars images:',len(cars_files))
print('Non-cars images:',len(noncars_files))

X_train=cars_files + noncars_files
y_train=np.array([1]*len(cars_files) + [0]*len(noncars_files))

val_size=0.2

batch_size = 500
nb_epoch = 50

generate=10

print(X_train[0])

X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=val_size,random_state=random_seed)

print('Train:',len(X_train),len(y_train))
print('Test:',len(X_val),len(y_val))

def transform_image(img,ang_range,shear_range,trans_range):
    '''
    This function transforms images to generate new images.
    The function takes in following arguments,
    1- Image
    2- ang_range: Range of angles for rotation
    3- shear_range: Range of values to apply affine transform to
    4- trans_range: Range of values to apply translations over.

    A Random uniform distribution is used to generate different parameters for transformation

    '''
    # Rotation

    ang_rot = np.random.uniform(ang_range)-ang_range/2
    rows,cols,ch = img.shape
    Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)

    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])

    # Shear
    pts1 = np.float32([[5,5],[20,5],[5,20]])

    pt1 = 5+shear_range*np.random.uniform()-shear_range/2
    pt2 = 20+shear_range*np.random.uniform()-shear_range/2

    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])

    shear_M = cv2.getAffineTransform(pts1,pts2)

    img = cv2.warpAffine(img,Rot_M,(cols,rows))
    img = cv2.warpAffine(img,Trans_M,(cols,rows))
    img = cv2.warpAffine(img,shear_M,(cols,rows))

    return img

def trainGenerator():
    while True:
        batchx=[]
        batchy=[]
        for iii,(fname,cl) in enumerate(list(zip(X_train,y_train))):
            img=cv2.imread(fname)
            for ggg in range(generate):
                img=transform_image(img,30,20,1)
                # cv2.imshow('gen',img_)
                # cv2.waitKey(0)
                processed=process_image(img)
                batchx.append(processed)
                batchy.append([cl])
                if len(batchx)==batch_size:
                    bx=np.array(batchx)
                    by=np.array(batchy)
                    batchx=[]
                    batchy=[]
                    yield bx,by

def valGenerator():
    while True:
        batchx=[]
        batchy=[]
        for fname,cl in zip(X_val,y_val):
            img=cv2.imread(fname)
            for ggg in range(generate):
                img=transform_image(img,30,20,1)
                processed=process_image(img)
                batchx.append(processed)
                batchy.append([cl])
                if len(batchx)==batch_size:
                    bx=np.array(batchx)
                    by=np.array(batchy)
                    batchx=[]
                    batchy=[]
                    yield bx,by

input_shape=process_image(np.zeros(shape=(64,64,3))).shape

print(input_shape, 'input shape')
for t in trainGenerator():
    print(t[0].shape,t[1].shape)
    break
kernel_size = (3,3)
model = Sequential()

model.add(Convolution2D(256, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(256, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(128, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(Convolution2D(128, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(Convolution2D(128, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
# model.add(Convolution2D(256, kernel_size[0], kernel_size[1]))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Convolution2D(512, kernel_size[0], kernel_size[1]))
# model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dense(1024))
model.add(Activation('relu'))
# model.add(Dense(1024))
# model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
# model.add(Activation('softmax'))

model.summary()

model.compile(loss='mse', optimizer=Adam())

history = model.fit_generator(trainGenerator(),
                    validation_data=valGenerator(),
                    samples_per_epoch=len(X_train)*generate,
                    nb_val_samples=len(X_val)*generate,
                    nb_epoch=nb_epoch,
                    verbose=1)



# Save model as json file
json_string = model.to_json()

with open('model.json', 'w') as outfile:
	json.dump(json_string, outfile)

	# save weights
	model.save_weights('./model.h5')
	print("Overwrite Successful")
