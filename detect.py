#!/usr/bin/python3
import sys
import json
import glob
import cv2
import numpy as np
import tensorflow as tf
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

np.random.seed(123456)

model = None

cl_list=sorted(glob.glob('more-examples/*/'))[:]
cl_list=[x.split('/')[-2] for x in cl_list]
print(cl_list)
def process_image(img):
    img=cv2.resize(img,(64,64))
    img=img.astype(np.uint8)
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
    # img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img/255.0-0.5

def detect(model,image_array):
    image_array=process_image(image_array)

    transformed_image_array = image_array.reshape(( 1,
                                           image_array.shape[0],
                                           image_array.shape[1],
                                           image_array.shape[2]))
    cl = model.predict(transformed_image_array, batch_size=1)
    return cl[0][0]

def loadNet():
    with open('model.json', 'r') as jfile:
        model = model_from_json(json.load(jfile))
        model.compile("adam", "mse")
        weights_file = 'model.h5'
        model.load_weights(weights_file)
        return model
