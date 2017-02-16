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

def showHistograms(img,label):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    luv = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    YCrCb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

    colorspaces={'R G B':img,'H S V':hsv,'L U V':luv,'H L S':hls,'Y U V':yuv,'Y Cr Cb':YCrCb}

    fig = plt.figure(figsize=(len(colorspaces.keys()),3))
    count=0
    for name,img in colorspaces.items():
        rhist = np.histogram(img[:,:,0], bins=32, range=(0, 256))
        ghist = np.histogram(img[:,:,1], bins=32, range=(0, 256))
        bhist = np.histogram(img[:,:,2], bins=32, range=(0, 256))

        bin_edges = rhist[1]
        bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2

        names=name.split(' ')

        plt.subplot2grid((len(colorspaces.keys()),3), (count,0))
        plt.bar(bin_centers, rhist[0])
        plt.xlim(0, 256)
        plt.title(names[0]+' Histogram')
        plt.subplot2grid((len(colorspaces.keys()),3), (count,1))
        plt.bar(bin_centers, ghist[0])
        plt.xlim(0, 256)
        plt.title(names[1]+' Histogram')
        plt.subplot2grid((len(colorspaces.keys()),3), (count,2))
        plt.bar(bin_centers, bhist[0])
        plt.xlim(0, 256)
        plt.title(names[2]+' Histogram')
        count=count+1
    plt.show()

# Define a function to return HOG features and visualization
def get_hog_features(img, vis=True, pix_per_cell = 3, cell_per_block = 2, orient = 2):
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
    return np.concatenate((spatial_features, hist_features))



cars_imgs,noncars_imgs=loadImages(cars,noncars)
car_features=[]
noncar_features=[]
cv2.imshow('noncar',noncars_imgs[0])
cv2.waitKey(0)
showHistograms(noncars_imgs[0],'noncar')

def get_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    features = get_hog_features(gray, vis=False, pix_per_cell = 2, cell_per_block = 2, orient = 8)
    # return features
    feat=extract_features(img, cspace='HSV', spatial_size=(8, 8), hist_bins=8, hist_range=(0, 256))
    return np.concatenate((features,feat))

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
    print('My SVC predicts: ', svc.predict(X_test[0:10]))
    print('For labels: ', y_test[0:10])

# Here is your draw_boxes function from the previous exercise
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

    # Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_windows = np.int(xspan/nx_pix_per_step) - 1
    ny_windows = np.int(yspan/ny_pix_per_step) - 1
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

image=cv2.imread('datasets/object-dataset/1478019952686311006.jpg')

windows0 = slide_window(image, x_start_stop=[None, None], y_start_stop=[500, 570],
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5))
windows1 = slide_window(image, x_start_stop=[None, None], y_start_stop=[470, 600],
                    xy_window=(128, 128), xy_overlap=(0.5, 0.5))
windows2 = slide_window(image, x_start_stop=[None, None], y_start_stop=[450,None],
                    xy_window=(256, 256), xy_overlap=(0.75, 0.5))
windows3 = slide_window(image, x_start_stop=[None, None], y_start_stop=[200,None],
                    xy_window=(512,512), xy_overlap=(0.5, 0.5))

windows=windows0+windows1+windows2+windows3


window_img = draw_boxes(image, windows1, color=(0, 0, 255), thick=4)
window_img = draw_boxes(window_img, windows2, color=(255, 0, 0), thick=3)
window_img = draw_boxes(window_img, windows3, color=(0, 255, 0), thick=2)
# window_img = draw_boxes(window_img, windows0, color=(255, 255, 0), thick=1)
cv2.imshow('result',window_img)
# cv2.waitKey(0)
# plt.show()



# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler):
    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in tqdm(windows):
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        cv2.imshow('test',test_img)
        cv2.waitKey(10)
        # 4)
        features = get_features(test_img)
        print('features.shape',features.shape)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape((1,-1)))
        #6) Predict using your classifier
        prediction = clf.predict(test_features.reshape((1,-1)))
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows

print('XXXX',image.shape)

for fimg in glob.glob('datasets/object-dataset/*.jpg'):
    print(fimg)
    image=cv2.imread(fimg)
    print(type(image))
    print(image.shape)
    hot_windows=search_windows(image,windows,svc,X_scaler)
    window_img = draw_boxes(image.copy(), hot_windows, color=(0, 0, 255), thick=6)
    cv2.imshow('hot',window_img)
    cv2.waitKey(0)
