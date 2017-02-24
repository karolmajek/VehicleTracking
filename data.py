#!/usr/bin/python3
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from skimage.feature import hog
from tqdm import tqdm
import numpy as np
import cv2
import glob
import json
import pickle


def loadImages():
    cars=glob.glob('datasets/vehicles/*/*.png')
    noncars=glob.glob('datasets/non-vehicles/*/*.png')

    print('Cars images:',len(cars))
    print('Non-cars images:',len(noncars))
    cars_imgs=[]
    noncars_imgs=[]
    try:
        data = pickle.load(open("cars-noncars-dataset.p", "rb"))
        print('Loading pickled images')
        cars_imgs=data['cars']
        noncars_imgs=data['noncars']
    except (OSError, IOError) as e:
        print('Loading images')
        for fname in tqdm(cars):
            cars_imgs.append(cv2.imread(fname))
        for fname in tqdm(noncars):
            noncars_imgs.append(cv2.imread(fname))
        print('Pickling images')
        data={'cars':cars_imgs,'noncars':noncars_imgs}
        pickle.dump(data, open("cars-noncars-dataset.p", "wb"))
    return cars_imgs,noncars_imgs
