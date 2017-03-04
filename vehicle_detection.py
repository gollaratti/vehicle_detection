# -*- coding: utf-8 -*-
"""
v1.0 - 27-Feb-2017
Vehicle detection - Initial version
"""

##############################################################################
### INCLUDES
##############################################################################
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from skimage.feature import hog
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
# NOTE: the next import is only valid 
# for scikit-learn version >= 0.18
# if you are using scikit-learn <= 0.17 then use this:
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
import pickle


##############################################################################
### Feature Extraction
##############################################################################
def bin_spatial(img, size=(32, 32)):
    '''
    This function computes binned color feaures in HSV. Input color is assumed
    to be in BGR format.
    '''
    # Use cv2.resize().ravel() to create the feature vector
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    # Return the feature vector
    return np.hstack((color1, color2, color3))

 
def color_hist(img, nbins=32, bins_range=(0, 256), visualize = False):
    '''
    This function computes the color histogram features
    '''
    
    hsv = cv2.cvtColor(np.copy(img), cv2.COLOR_RGB2HSV)
    
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(hsv[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(hsv[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(hsv[:,:,2], bins=nbins, range=bins_range)
    # visualize color_histogram if required
    if (visualize == True):
        # Generating bin centers
        bin_edges = channel1_hist[1]
        bincen = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
        fig = plt.figure(figsize=(12,3))
        plt.subplot(131)
        plt.bar(bincen, channel1_hist[0])
        plt.xlim(0, 256)
        plt.title('H Histogram')
        plt.subplot(132)
        plt.bar(bincen, channel2_hist[0])
        plt.xlim(0, 256)
        plt.title('S Histogram')
        plt.subplot(133)
        plt.bar(bincen, channel3_hist[0])
        plt.xlim(0, 256)
        plt.title('V Histogram')
        fig.tight_layout()
        plt.savefig('./output_images/histogram.jpg')
        plt.close()
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features
    
def hog_feature(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True, hog_channel='ALL', visualize = False):
    '''
    This function extracts the hog feature for one or al 3 channels.
    '''
    if hog_channel == 'ALL':
        hog_features = []
        for channel in range(img.shape[2]):
            hog_features.extend(hog(img[:,:,channel], orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), 
                                  cells_per_block=(cell_per_block, cell_per_block), visualise=vis, feature_vector=feature_vec))      
    else:
        if(visualize == True):
            
            hog_features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), 
                                  cells_per_block=(cell_per_block, cell_per_block), visualise=True, feature_vector=feature_vec)
            return hog_features,hog_image
        else:
            hog_features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), 
                                  cells_per_block=(cell_per_block, cell_per_block), visualise=vis, feature_vector=feature_vec)
             
    
    return hog_features


def grad_mag(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # 0) Convert to Gray image
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 1) Take the derivative in x and y 
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1,0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0,1, ksize=sobel_kernel)
    sobel = np.sqrt(np.square(sobelx) + np.square(sobely))

    # 2) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*sobel/np.max(sobel))
    # 3) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    mag_binary = np.zeros_like(scaled_sobel)
    mag_binary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    # 4) Return this mask  
    return mag_binary
    

def extract_features(imgs, cspace='HSV', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256), orient = 9, pix_per_cell = 8,  cell_per_block = 2, visualize = False):
    '''
    This function extracts the features from a list of images
    '''
    # Create a list to append feature vectors to
    features = []
    if(visualize == False):
        # Iterate through the list of images
        for file in imgs:
            # Read in each one by one range: 0 to 255 for .png images
            image = cv2.imread(file)
            # Convert the image to RGB color space from BGR color space 
            feature_image = (cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            #Collect spatial fetures
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            # Apply color_hist() also with a color space option now
            hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range, visualize = False)
            #Apply hog_feature() 
            hog_features = hog_feature(feature_image,orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True, hog_channel='ALL')
            #apply gradient magnitude feature
            grad_features = grad_mag(feature_image, sobel_kernel=9, mag_thresh=(20, 255))
            # Append the new feature vector to the features list
            features.append(np.concatenate((spatial_features, hist_features, hog_features, grad_features.ravel())))
    else:
        # Read the file
        image = cv2.imread('./vehicles/KITTI_extracted/1141.png')
        # Convert the image to HSV color space from BGR color space 
        feature_image =  (cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32))
        # Apply bin_spatial() to get spatial color features
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # Apply color_hist() also with a color space option now
        hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range, visualize = True)
        #Apply hog_feature() 
        hog_features, hog_image = hog_feature(feature_image[:,:,0],orient, pix_per_cell, cell_per_block, vis=True, feature_vec=True, hog_channel=1, visualize = True)
        fig = plt.figure()
        plt.subplot(121)
        plt.imshow(image, cmap='gray')
        plt.title('Example Car Image')
        plt.subplot(122)
        plt.imshow(hog_image, cmap='gray')
        plt.title('HOG H channel Visualization')
        plt.savefig('./output_images/hog_h.jpg')
        plt.close()
        
        hog_features, hog_image = hog_feature(feature_image[:,:,1],orient, pix_per_cell, cell_per_block, vis=True, feature_vec=True, hog_channel=1, visualize = True)
        fig = plt.figure()
        plt.subplot(121)
        plt.imshow(image, cmap='gray')
        plt.title('Example Car Image')
        plt.subplot(122)
        plt.imshow(hog_image, cmap='gray')
        plt.title('HOG S Channel Visualization')
        plt.savefig('./output_images/hog_s.jpg')
        plt.close()
        
        
        hog_features, hog_image = hog_feature(feature_image[:,:,2],orient, pix_per_cell, cell_per_block, vis=True, feature_vec=True, hog_channel=1, visualize = True)
        fig = plt.figure()
        plt.subplot(121)
        plt.imshow(image, cmap='gray')
        plt.title('Example Car Image')
        plt.subplot(122)
        plt.imshow(hog_image, cmap='gray')
        plt.title('HOG V Channel Visualization')
        plt.savefig('./output_images/hog_v.jpg')
        plt.close()
        # Append the new feature vector to the features list
        features.append(np.concatenate((spatial_features, hist_features, hog_features)))
        
        plt.hist(features)
        
    # Return list of feature vectors
    return features

##############################################################################
### Train & test the classifier
##############################################################################
def process_features():
    '''
    This function extracts the training data for features and returns 
    X_train, X_test, y_train, y_test for training the model and 
    X_scaler for scaling prediction data
    '''
    cars = []
    notcars = []
    # Read in car images
    images = glob.glob('./database/vehicles/*.png')
    for image in images:
        cars.append(image)
    
    # Read in non-car images
    images = glob.glob('./database/non-vehicles/*.png')
    for image in images:
        notcars.append(image)
           
    #set tuned feature collection values
    spatial = 16
    histbin = 32
    # HOG parameters
    orient_value = 9
    pix_per_cell_value = 8
    cell_per_block_value = 2
    
    #visualization
    vis = False    
    
    if(vis == True):
    
        car_features = extract_features('./vehicles/KITTI_extracted/1141.png', cspace='HSV', spatial_size=(spatial, spatial),
                            hist_bins=histbin, hist_range=(0, 256), orient = orient_value,
                            pix_per_cell = pix_per_cell_value,  cell_per_block = cell_per_block_value, visualize = vis)
        notcar_features = extract_features('./vehicles/KITTI_extracted\1141.png', cspace='HSV', spatial_size=(spatial, spatial),
                            hist_bins=histbin, hist_range=(0, 256), orient = orient_value,
                            pix_per_cell = pix_per_cell_value,  cell_per_block = cell_per_block_value, visualize = vis)
    else:
        car_features = extract_features(cars, cspace='HSV', spatial_size=(spatial, spatial),
                            hist_bins=histbin, hist_range=(0, 256), orient = orient_value,
                            pix_per_cell = pix_per_cell_value,  cell_per_block = cell_per_block_value, visualize = False)
        notcar_features = extract_features(notcars, cspace='HSV', spatial_size=(spatial, spatial),
                            hist_bins=histbin, hist_range=(0, 256), orient = orient_value,
                            pix_per_cell = pix_per_cell_value,  cell_per_block = cell_per_block_value, visualize = False)
        
        
    
    
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    
    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    
    
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)
    
    return X_train, X_test, y_train, y_test, X_scaler
 
    
def train_classifier(X_train, X_test, y_train, y_test):
    '''
    This function trains and tests the classifier and saves the trained classifier
    '''
    
    print('Number of training samples: ', X_train.shape[0])
    print('Number of validation samples: ', X_test.shape[0])
    print('Number of cars in training samples: ', np.count_nonzero(y_train))
    
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC 
    svc = LinearSVC(random_state = 100, verbose=True, max_iter=5000)

    # Check the training time for the SVC
    t=time.time()
    #clf.fit(X_train, y_train)
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
    
    return svc



##############################################################################
### Vehicle Detection Pipeline for Images
##############################################################################

def find_cars(img, ystart, ystop, xstart, xstop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    ''' 
    This function extracts features from the image and finds vehicles in the
    image and return the bounding boxes.
    '''

    img = img.astype(np.float32)
    bbox_list = []
    
    img_tosearch = img[ystart:ystop,xstart:xstop,:]
    ctrans_tosearch = img_tosearch 

    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1 

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1 
    cells_per_step = 1  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    #hog_features = np.hstack(hog_feature())
    hog1 = hog_feature(ch1, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False, hog_channel=0)
    hog2 = hog_feature(ch2, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False, hog_channel=1)
    hog3 = hog_feature(ch3, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False, hog_channel=2)
    

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch

            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)
            grad_features = grad_mag(subimg, sobel_kernel=9, mag_thresh=(20, 255)).ravel()
   
            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack(( spatial_features,hist_features, hog_features, grad_features)).reshape(1, -1))      
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale+xstart)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                bbox_list.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
    return bbox_list
    

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,255,0), 2)
    # Return the image
    return img

##############################################################################
### Vehicle Detection Pipeline for Video frames
##############################################################################

# Define a class to receive the characteristics of  building box detection
class Bboxes():
    def __init__(self):

        self.bbox_list1 = []

    def get_bboxes(self, bboxl1):                               
        self.bbox_list1.append(bboxl1)

        # pop out the oldest bboxes over 5 frames
        if(len(self.bbox_list1) >=6):
            _ = self.bbox_list1.pop(0)

              
        return self.bbox_list1

def process_frame(img):
    '''
    This function processes each frame of the video pipeline
    '''
    global bbox, svc, X_scaler, count
    ystart = 400
    ystop = 600
    xstart = 700
    xstop = 1280
    orient = 9
    pix_per_cell = 8 
    cell_per_block = 2
    spatial_size = (16,16)
    hist_bins = 32
  
    scale = 1
    bbox1 = find_cars(img, ystart, ystop, xstart, xstop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
   
    bbox_list1 = bbox.get_bboxes(bbox1)
#    
#    # Read in image similar to one given
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
#    
#    # Add heat to each box in box list over 5 frames or lesser in the beginning
    num = len(bbox_list1)

    for i in range(len(bbox_list1)):
        heat = add_heat(heat,bbox_list1[i])
    
   
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 20*num) 
    
    # Visualize the heatmap when diplaying    
    heatmap = np.clip(heat, 0, 255)
    
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
  
    
    return draw_img
    
##############################################################################
### Running the Pipelines
##############################################################################

# define bounding box object
bbox = Bboxes()

#prepare and process feature vectors
X_train, X_test, y_train, y_test, X_scaler  = process_features()

#train the classifier    
svc = train_classifier(X_train, X_test, y_train, y_test)

#save the classifier
pickle.dump(svc, open('svc.dat','wb'))
pickle.dump(X_scaler, open('scaler.dat','wb'))

#load the classifier
svc = pickle.load(open('svc.dat','rb'))
X_scaler = pickle.load(open('scaler.dat','rb'))

#run project video
output_video = 'out_project_video.mp4'
clip1 = VideoFileClip("project_video.mp4")
output_clip = clip1.fl_image(process_frame)
output_clip.write_videofile(output_video, audio=False)