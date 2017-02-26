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
import time

from sklearn.preprocessing import StandardScaler
from data import loadImages
from hog import *
from scipy.ndimage.measurements import label


cars_imgs,noncars_imgs=loadImages()
car_features=[]
noncar_features=[]

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
        pred=svc.predict(X_scaler.transform(get_features(test_img).reshape(1,-1)))
        # print(prediction,pred)
        #7) If positive (prediction == 1) then save the window
        if pred >0.0:
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
    # fig = plt.figure(figsize=(12,4))
    # plt.subplot(131)
    # plt.imshow(cars_imgs[car_ind])
    # plt.title('Original Image')
    # plt.subplot(132)
    # plt.plot(X[car_ind])
    # plt.title('Raw Features')
    # plt.subplot(133)
    # plt.plot(scaled_X[car_ind])
    # plt.title('Normalized Features')
    # fig.tight_layout()
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
# image=cv2.imread('CarND-Vehicle-Detection/test_images/test1.jpg')

def generate_windows(image,car_pos):
    generate_windows.counter+=1
    windows0=[]
    windows1=[]
    windows2=[]
    windows3=[]
    if len(car_pos)==0 or generate_windows.counter>10:
        generate_windows.counter=0
        # windows0 = slide_window(image, x_start_stop=[200, 1000], y_start_stop=[400, 600], xy_window=(64, 64), xy_overlap=(0.75, 0.5))
        # windows3 = slide_window(image, x_start_stop=[None, None], y_start_stop=[200,None], xy_window=(256,256), xy_overlap=(0.5, 0.5))
        windows2 = slide_window(image, x_start_stop=[None, None], y_start_stop=[400,680], xy_window=(96, 96), xy_overlap=(0.5, 0.25))
        windows3 = slide_window(image, x_start_stop=[None, None], y_start_stop=[370, 680], xy_window=(128, 128), xy_overlap=(0.2, 0.1))
    if len(car_pos)>0:
        for bbox in car_pos:
            xmin=bbox[0][0]-30
            xmax=bbox[1][0]+30
            ymin=bbox[0][1]
            ymax=bbox[1][1]

            xmin=max(0,xmin)
            xmax=min(image.shape[1],xmax)
            ymin=max(0,ymin)
            ymax=min(image.shape[0],ymax)

            windows0 += windows0 + slide_window(image, x_start_stop=[xmin, xmax], y_start_stop=[ymin, ymax], xy_window=(48,48), xy_overlap=(0.2, 0.2))
            windows1 += windows1 + slide_window(image, x_start_stop=[xmin, xmax], y_start_stop=[ymin, ymax], xy_window=(64, 64), xy_overlap=(0.75, 0.75))
            windows2 += windows2 +  slide_window(image, x_start_stop=[xmin, xmax], y_start_stop=[ymin,ymax], xy_window=(96, 96), xy_overlap=(0.75, 0.75))
            windows3 += windows3 + slide_window(image, x_start_stop=[xmin, xmax], y_start_stop=[ymin, ymax], xy_window=(128,128), xy_overlap=(0.2, 0.2))
    windows=windows0+windows1+windows2+windows3


    window_img = draw_boxes(image, windows1, color=(0, 0, 255), thick=4)
    window_img = draw_boxes(window_img, windows2, color=(255, 0, 0), thick=3)
    window_img = draw_boxes(window_img, windows3, color=(0, 255, 0), thick=2)
    window_img = draw_boxes(window_img, windows0, color=(0, 255, 255), thick=1)
    # cv2.imshow('result',window_img)
    # cv2.waitKey(0)
    return windows,window_img
generate_windows.counter=0

times=[]

for fimg in sorted(glob.glob('CarND-Vehicle-Detection/test_images/*.jpg')):
    print(fimg)
    img=cv2.imread(fimg)
    img=cv2.resize(img,(1280,720))
    start = time.time()
    windows,win_img=generate_windows(img,[])

    hot_windows=search_windows(img,windows,svc,X_scaler)
    window_img = draw_boxes(img.copy(), hot_windows, color=(255, 255, 255), thick=1)

    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    # Add heat to each box in box list
    heat = add_heat(heat,hot_windows)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,0)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img,car_pos = draw_labeled_bboxes(np.copy(window_img), labels)

    end = time.time()

    mix=np.concatenate((np.concatenate((img.astype(np.uint8),draw_img.astype(np.uint8)),axis=0),np.concatenate((win_img.astype(np.uint8),np.dstack([heatmap,heatmap,heatmap]).astype(np.uint8)),axis=0)),axis=1)

    end = time.time()
    times.append((end-start,len(car_pos)))

    cv2.imwrite('images/'+fimg.split('/')[-1],mix)

plt.plot(times)
plt.ylabel('seconds/detected cars')
plt.xlabel('frames')
plt.legend(('Frame time in seconds','Detected cars'))
plt.savefig('images/img-det.png')



cap = cv2.VideoCapture('CarND-Vehicle-Detection/project_video.mp4')

last_car_pos=[]
times=[]
while(cap.isOpened()):
    ret, img = cap.read()
    if ret and not img is None:
        start = time.time()

        windows,win_img=generate_windows(img,last_car_pos)

        hot_windows=search_windows(img,windows,svc,X_scaler)
        window_img = draw_boxes(img.copy(), hot_windows, color=(255, 255, 255), thick=1)

        heat = np.zeros_like(img[:,:,0]).astype(np.float)
        # Add heat to each box in box list
        heat = add_heat(heat,hot_windows)

        # Apply threshold to help remove false positives
        heat = apply_threshold(heat,0)

        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        draw_img,car_pos = draw_labeled_bboxes(np.copy(window_img), labels)
        last_car_pos=car_pos

        end = time.time()

        times.append((end-start,len(car_pos)))

        mix=np.concatenate((np.concatenate((img.astype(np.uint8),draw_img.astype(np.uint8)),axis=0),np.concatenate((win_img.astype(np.uint8),np.dstack([heatmap,heatmap,heatmap]).astype(np.uint8)),axis=0)),axis=1)

        cv2.imwrite('output-video/%08d.jpg'%len(times),mix)
    else:
        break
# print(times)
plt.plot(times)
plt.ylabel('seconds/detected cars')
plt.xlabel('frames')
plt.legend(('Frame time in seconds','Detected cars'))
plt.savefig('images/detections.png')
# plt.show()
cap.release()
