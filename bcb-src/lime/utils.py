#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 13:08:44 2018

@author: mohebbi
"""
import cv2
import os
import glob
import shutil
from random import sample, randint, shuffle
import numpy as np
import pandas as pd
from sklearn import metrics
from PIL import Image

# plots
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img



def preprocess_images(filename, dest_folder, img_dimensions=(64, 64)):
    # function to pre-process images and create training and test sets. 
    src = os.path.join('crater_data', 'images', filename)
    dst = os.path.join('crater_data', dest_folder)
    tgt_height, tgt_width = img_dimensions

    # create new directories if necessary
    for imgtype in ['crater', 'non-crater']:
        tgdir = os.path.join(dst, imgtype)
        if not os.path.isdir(tgdir):
            os.makedirs(tgdir)

    for src_filename in glob.glob(os.path.join(src, '*', '*.jpg')):
        #print(src_filename)
        pathinfo = src_filename.split(os.path.sep)
        img_type = pathinfo[-2] # crater or non-crater
        filename = pathinfo[-1] # the actual name of the jpg

        dst_filename = os.path.join(dst, img_type, filename)

        # read the original image and get size info
        src_img = cv2.imread(src_filename)

        # resize image, normalize and write to disk
        scaled_img = cv2.resize(src_img, (tgt_height, tgt_width))
        cv2.normalize(scaled_img, scaled_img, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(dst_filename, scaled_img)
    
    print(dest_folder + " Done!")
    
def move_random_files(path_from, path_to, n):
    # function for moving random files from one directory to another (used for creating train and test set)
    files = os.listdir(path_from)
    files.sort()
    files = files[1:] #omiting .DS_Store

    for i in sample(range(0, len(files)-1), n):
        f = files[i]
        src = path_from + f
        dst = path_to + f
        shutil.move(src, dst)
        
def preview_random_image(path):
    # function for previewing a random image from a given directory
    files = os.listdir(path)
    files.sort()
    img_name = files[randint(1, len(files) - 1)]
    img_preview_name = path + img_name
    image = Image.open(img_preview_name)
    plt.imshow(image)
    plt.title(img_name)
    plt.show()
    width, height = image.size
    print ("Dimensions:", image.size, "Total pixels:", width * height)
    
def pretty_cm(y_pred, y_truth, labels, save_path):
    # pretty implementation of a confusion matrix
    cm = metrics.confusion_matrix(y_truth, y_pred)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, fmt="d", linewidths=.5, square = True, cmap = 'BuGn_r')
    # labels, title and ticks
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('Actual label')
    ax.set_title('Accuracy: {0}'.format(metrics.accuracy_score(y_truth, y_pred)), size = 15) 
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
    plt.savefig(save_path, dpi=400)
    
    
def img_to_1d_greyscale(img_path, size):
    # function for loading, resizing and converting an image into greyscale
    # used for logistic regression
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, size)
    return(pd.Series(img.flatten()))

def show_image(image):
    # function for viewing an image
    fig = plt.figure(figsize = (5, 25))
    ax = fig.add_subplot(111)
    ax.imshow(image, interpolation='none')
    plt.show()

def transform_image(path, size):
    # function for transforming images into a format supported by CNN
    x = load_img(path, target_size=(size[0], size[1]))
    x = img_to_array(x) / 255
    x = np.expand_dims(x, axis=0)
    return (x)
    
def evaluation_indices(y_pred, y_test):
    # function for getting correctly and incorrectly classified indices
    index = 0
    correctly_classified_indices = []
    misclassified_indices = []
    for label, predict in zip(y_test, y_pred):
        if label != predict: 
            misclassified_indices.append(index)
        else:
            correctly_classified_indices.append(index)
        index +=1
    return (correctly_classified_indices, misclassified_indices)