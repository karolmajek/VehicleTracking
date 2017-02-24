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

from sklearn.preprocessing import StandardScaler
from detect import detect,loadNet
from data import loadImages
from hog import *

model=loadNet()
model.summary()

cars_imgs,noncars_imgs=loadImages()
car_features=[]
noncar_features=[]
# cv2.imshow('noncar',noncars_imgs[0])
# cv2.waitKey(0)
# showHistograms(noncars_imgs[0],'noncar')

# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler):
    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        # 4)
        # prediction=detect(model,test_img)
        pred=svc.predict(X_scaler.transform(get_features(test_img).reshape(1,-1)))
        # print(prediction,pred)
        #7) If positive (prediction == 1) then save the window
        if pred >0.8:
            # cv2.imshow('test',test_img)
            # cv2.waitKey(1)
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows

try:
    data = pickle.load(open("cars-noncars-features.p", "rb"))
    print('Loading pickled data')
    car_features=data['car_features']
    noncar_features=data['noncar_features']
except (OSError, IOError) as e:
    print('Generating features')
    for img in tqdm(cars_imgs):
        features = get_features(img)
        car_features.append(features)
    for img in tqdm(noncars_imgs):
        features = get_features(img)
        noncar_features.append(features)

    car_features=np.array(car_features)
    noncar_features=np.array(noncar_features)

    print('Pickling data')
    data={'car_features':car_features,'noncar_features':noncar_features}
    pickle.dump(data, open("cars-noncars-features.p", "wb"))



if len(car_features) > 0:
    # Create an array stack of feature vectors

    X = np.vstack((car_features, noncar_features)).astype(np.float64)
    y = np.hstack((np.ones(len(car_features)),np.zeros(len(noncar_features))))

    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)


    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    car_ind = np.random.randint(0, len(cars_imgs))

    # Plot an example of raw and scaled features
    fig = plt.figure(figsize=(12,4))
    plt.subplot(131)
    plt.imshow(cars_imgs[car_ind])
    plt.title('Original Image')
    plt.subplot(132)
    plt.plot(X[car_ind])
    plt.title('Raw Features')
    plt.subplot(133)
    plt.plot(scaled_X[car_ind])
    plt.title('Normalized Features')
    fig.tight_layout()
    # plt.show()


    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('X_train',X_train.shape)
    print('y_train',y_train.shape)
    print('X_test',X_test.shape)
    print('y_test',y_test.shape)
    print('Feature vector length:', len(X_train[0]))

    # Use a linear SVC (support vector classifier)
    svc = LinearSVC()
    # Train the SVC
    print('Training...')
    svc.fit(X_train, y_train)

    print('Test Accuracy of SVC = ', svc.score(X_test, y_test))
    print('My SVC predicts: ', svc.predict(X_test))
    print('For labels: ', y_test)

# image=cv2.imread('datasets/object-dataset/1478019952686311006.jpg')
image=cv2.imread('CarND-Vehicle-Detection/test_images/test1.jpg')

print(image.shape)
windows0 = slide_window(image, x_start_stop=[None, None], y_start_stop=[400, None],
                    xy_window=(64, 64), xy_overlap=(0.75, 0.75))
windows1 = slide_window(image, x_start_stop=[None, None], y_start_stop=[370, None],
                    xy_window=(128, 128), xy_overlap=(0.75, 0.75))
windows2 = slide_window(image, x_start_stop=[None, None], y_start_stop=[450,None],
                    xy_window=(256, 256), xy_overlap=(0.75, 0.5))
windows3 = slide_window(image, x_start_stop=[None, None], y_start_stop=[200,None],
                    xy_window=(512,512), xy_overlap=(0.5, 0.5))

# windows=windows0+windows1+windows2+windows3
windows=windows0


window_img = draw_boxes(image, windows1, color=(0, 0, 255), thick=4)
window_img = draw_boxes(window_img, windows2, color=(255, 0, 0), thick=3)
window_img = draw_boxes(window_img, windows3, color=(0, 255, 0), thick=2)
window_img = draw_boxes(window_img, windows0, color=(0, 255, 255), thick=1)
cv2.imshow('result',window_img)
# cv2.waitKey(0)

# for fimg in glob.glob('datasets/object-dataset/*.jpg'):
#     print(fimg)
#     image=cv2.imread(fimg)
#     print(type(image))
#     print(image.shape)
#     hot_windows=search_windows(image,windows,svc,X_scaler)
#     window_img = draw_boxes(image.copy(), hot_windows, color=(0, 0, 255), thick=6)
#     cv2.imshow('hot',window_img)
#     cv2.waitKey(0)



cap = cv2.VideoCapture('CarND-Vehicle-Detection/project_video.mp4')
for i in range(400):
    ret, img = cap.read()

while(cap.isOpened()):
    ret, img = cap.read()
    if not img is None:
        # cv2.imshow('video',img)
        hot_windows=search_windows(img,windows,svc,X_scaler)
        window_img = draw_boxes(img.copy(), hot_windows, color=(0, 0, 255), thick=6)
        cv2.imshow('hot',window_img)
        cv2.waitKey(1)
cap.release()
