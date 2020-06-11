import os
import cv2 
import glob
import random
import numpy as np

def load_crater_data(dataset_type):
    
    # set origin path for the images
    src = os.path.join('crater_data', 'samples', dataset_type, 'normalized_images')
    
    # this dict helps to create binary labels for the pictures
    labels_dict = {'crater': 1, 'non-crater': 0}
    
    images = []
    labels = []
    hot_one = []
    # get all images file paths
    for src_filename in glob.glob(os.path.join(src, '*', '*.png')):
        # extract info from file path
        pathinfo = src_filename.split(os.path.sep)
        img_type = pathinfo[-2] # crater or non-crater
        filename = pathinfo[-1] # the actual name of the jpg
        
        # read the grayscale version of the image, 
        # and normalize its values to be between 0 and 1
        img = cv2.imread(src_filename, cv2.IMREAD_GRAYSCALE) / 255.0
        
        # reshape the data structure to be a 1-D column vector
        img = img.flatten()
        
        # include the image data and its label into the sample list
        images.append(img)
        labels.append(labels_dict[img_type])
        hot_one.append([int(i==labels_dict[img_type]) for i in range(2)])
    
    # We have to shuffle the order before splitting between training data
    # and test data
    # will shuffle on next step
    #random.shuffle(samples)
    
    # determine slices for training and test data
    #splitpos = int(len(samples) * 0.7)
    #return samples[:splitpos], samples[splitpos:]
    
    # Will split data after this step. Return a single data set
    return np.array(images), np.array(labels), np.array(hot_one)

def load_crater_data_wrapper(dataset_type):
    
    # set origin path for the images
    src = os.path.join('crater_data', 'samples',dataset_type, 'normalized_images')
    
    # this dict helps to create binary labels for the pictures
    labels_dict = {'crater': 1, 'non-crater': 0}
    
    samples = []
    # get all images file paths
    for src_filename in glob.glob(os.path.join(src, '*', '*.png')):
        # extract info from file path
        pathinfo = src_filename.split(os.path.sep)
        img_type = pathinfo[-2] # crater or non-crater
        filename = pathinfo[-1] # the actual name of the jpg
        
        # read the grayscale version of the image, 
        # and normalize its values to be between 0 and 1
        img = cv2.imread(src_filename, cv2.IMREAD_GRAYSCALE) / 255.0
        
        # reshape the data structure to be a 1-D column vector
        img = img.flatten().reshape((len(img)**2, 1))
        
        # include the image data and its label into the sample list
        samples.append((img, labels_dict[img_type]))
    
    # We have to shuffle the order before splitting between training data
    # and test data
    random.shuffle(samples)
    
    # determine slices for training and test data
    splitpos = int(len(samples) * 0.7)
    return samples[:splitpos], samples[splitpos:]
    