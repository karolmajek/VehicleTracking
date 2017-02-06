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


cars=glob.glob('datasets/vehicles/*/*.png')
noncars=glob.glob('datasets/non-vehicles/*/*.png')

print('Cars images:',len(cars))
print('Non-cars images:',len(noncars))

def loadImages(cars,noncars):
    cars_imgs=[]
    noncars_imgs=[]
    try:
        data = pickle.load(open("cars-noncars-dataset.p", "rb"))
        print('Loading pickled data')
        cars_imgs=data['cars']
        noncars_imgs=data['noncars']
    except (OSError, IOError) as e:
        print('Loading images')
        for fname in tqdm(cars):
            cars_imgs.append(cv2.imread(fname))
        for fname in tqdm(noncars):
            noncars_imgs.append(cv2.imread(fname))
        print('Pickling data')
        data={'cars':cars_imgs,'noncars':noncars_imgs}
        pickle.dump(data, open("cars-noncars-dataset.p", "wb"))
    return cars_imgs,noncars_imgs

# Define a function to return HOG features and visualization
def get_hog_features(img, vis=True):
    pix_per_cell = 3
    cell_per_block = 2
    orient = 2
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), cells_per_block=(cell_per_block, cell_per_block), visualise=True, feature_vector=False)
        # Use skimage.hog() to get both features and a visualization
        return features, hog_image
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), cells_per_block=(cell_per_block, cell_per_block), visualise=False, feature_vector=True)
        return features


def bin_spatial(img, color_space='RGB', size=(32, 32)):
    # Convert image to new color space (if specified)
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(img)
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(feature_image, size).ravel()
    # Return the feature vector
    return features

def color_hist(image, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the RGB channels separately
    rhist = np.histogram(image[:,:,0], bins=32, range=(0, 256))
    ghist = np.histogram(image[:,:,1], bins=32, range=(0, 256))
    bhist = np.histogram(image[:,:,2], bins=32, range=(0, 256))
    # Generating bin centers
    bin_edges = rhist[1]
    bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
    # Return the individual histograms, bin_centers and feature vector
    # return rhist, ghist, bhist, bin_centers, hist_features
    return hist_features

def extract_features(image, cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256)):
    # apply color conversion if other than 'RGB'
    if cspace != 'RGB':
        if cspace == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    else:
        feature_image = np.copy(image)
    # Apply bin_spatial() to get spatial color features
    spatial_features = bin_spatial(feature_image, size=spatial_size)
    # Apply color_hist() also with a color space option now
    hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
    # Append the new feature vector to the features list
    # print('#'*50)
    # print(spatial_features.shape)
    # print(hist_features.shape)
    # print('#'*50)
    return np.concatenate((spatial_features, hist_features))



cars_imgs,noncars_imgs=loadImages(cars,noncars)
car_features=[]
noncar_features=[]



try:
    data = pickle.load(open("cars-noncars-features.p", "rb"))
    print('Loading pickled data')
    car_features=data['car_features']
    noncar_features=data['noncar_features']
except (OSError, IOError) as e:
    print('Generating features')
    for img in tqdm(cars_imgs):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # features, hog_image = get_hog_features(gray)
        # cv2.imshow('hog',cv2.resize(hog_image,(200,200)))
        # cv2.waitKey(1)
        features = get_hog_features(gray,False)
        feat=extract_features(img, cspace='RGB', spatial_size=(32, 32), hist_bins=32, hist_range=(0, 256))
        # print(features.dtype)
        # print(feat.dtype)
        car_features.append(np.concatenate((features,feat)))
        # print(feat.shape)
        # print(features.shape)
    for img in tqdm(noncars_imgs):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # features, hog_image = get_hog_features(gray)
        # cv2.imshow('hog',cv2.resize(hog_image,(200,200)))
        # cv2.waitKey(1)
        features = get_hog_features(gray,False)
        feat=extract_features(img, cspace='RGB', spatial_size=(32, 32), hist_bins=32, hist_range=(0, 256))
        noncar_features.append(np.concatenate((features,feat)))
        # print(feat.shape)
        # print(features.shape)
    car_features=np.array(car_features)
    noncar_features=np.array(noncar_features)

    print('Pickling data')
    data={'car_features':car_features,'noncar_features':noncar_features}
    pickle.dump(data, open("cars-noncars-features.p", "wb"))





if len(car_features) > 0:
    # Create an array stack of feature vectors

    X = np.vstack((car_features, noncar_features)).astype(np.float64)
    y = np.hstack((np.ones(len(car_features)),np.zeros(len(noncar_features))))

    print(X.shape,X.dtype)
    print(y.shape,y.dtype)

    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)


    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    car_ind = np.random.randint(0, len(cars))

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
    print(X_test.shape)
    print(X_test[0:10].shape)
    print('My SVC predicts: ', svc.predict(X_test[0:10]))
    print('For labels: ', y_test[0:10])
