#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 12:27:53 2019

@author: mohebbi
"""

from skimage.transform import pyramid_gaussian
import cv2 as cv
from helper import sliding_window
import os
import csv
from crater_cnn import Network as CNN
from crater_nn import Network as NN
import Param
import numpy as np
import seaborn as sns

cwd = os.getcwd()
# This script will go through all image tiles and detects crater area using sliding window method. This script use the output of FCN Segmentation code as input of this step.
# for every crater candidate, we look at the binary mask of segmentation to remove non-crater area. We consider a candidate as a potential crater area, if more than 50 (default threshold) percent of its area was creater. 
# Then, write results as a csv file to the results folder. The results of this script is the input to the remove_duplicates.py script. 

# Task: add the part that we do calculations for a range of threshold values and save a plot about  it.

# input : a 2D of int (segmentation output) 
# output: a colored image of input
def give_color_to_seg_img(seg,n_classes):
    # generate a color image based on the segmented image.
    
    if len(seg.shape)==3:
        seg = seg[:,:,0]
    seg_img = np.zeros( (seg.shape[0],seg.shape[1],3) ).astype('float')
    colors = sns.color_palette("hls", n_classes)
    
    for c in range(n_classes):
        segc = (seg == c)
        seg_img[:,:,0] += (segc*( colors[c][0] ))
        seg_img[:,:,1] += (segc*( colors[c][1] ))
        seg_img[:,:,2] += (segc*( colors[c][2] ))

    return(seg_img, colors)
    
# it does reverse of the above function
def get_seg_from_img(seg_img, colors, n_classes = 2):
    
    seg_mtx = np.zeros( (seg_img.shape[0],seg_img.shape[1]) , dtype=int)
    
    for c in range(n_classes):
        segc = (colors[c] == seg_img)
        seg_mtx += segc.astype(np.int)

    return(seg_mtx)
    
# This function determines if the window area, is a potential crater area or not. 
# This function calculates the crater score measure for for each window size (potential crater area). This measure is the number of 1 pixels to size of the window.
# This function resturns true if crater score is bigger than equal to threshold.
# The input is the loaded binary mask which is the output of FCN Segmentation phase. 
def is_potential_crater(bin_mask,x,y,resized_w, windowSize, threshold = 0.5):
    
    windowSize_b = int((windowSize[1] * bin_mask.shape[1]) / resized_w + 1) # we consider scaling down the image for pyramid sliding window for binary mask too. 
    window = bin_mask[y:y + windowSize_b, x:x + windowSize_b]
    crater_score = float(np.sum(window)) / (windowSize[1] * windowSize[0])
    answ = crater_score >= threshold
    return answ, crater_score


if __name__ == "__main__":
    
    param = Param.Param()
    crater_threshold = 0.5
    n_classes= 2
    # setup CNN
    cnn = CNN(img_shape=(50, 50, 1))
    cnn.add_convolutional_layer(5, 16)
    cnn.add_convolutional_layer(5, 36)
    cnn.add_flat_layer()
    cnn.add_fc_layer(size=64, use_relu=True)
    cnn.add_fc_layer(size=16, use_relu=True)
    cnn.add_fc_layer(size=2, use_relu=False)
    cnn.finish_setup()
    # model.set_data(data)
    
    # restore previously trained CNN model
    cnn_model_path = os.path.join(cwd, 'models/cnn/crater_model_cnn.ckpt')
    cnn.restore(cnn_model_path)
        
    # go through all the tile folders
    gt_list = ["1_24", "1_25", "2_24", "2_25", "3_24", "3_25"]
    #gt_list = ["1_24"]
    
    for gt_num in gt_list:
    
        tile_img = 'tile' + gt_num
        print("Working on " + tile_img)
        
        path = os.path.join('crater_data', 'tiles')
        bin_mask_path = os.path.join('crater_data', 'FCN-output', tile_img + '.csv')
        
        img = cv.imread(os.path.join(path, tile_img + '.pgm'), 0)
        img = cv.normalize(img, img, 0, 255, cv.NORM_MINMAX)/255.0
        
        # load the output of Gen_Mask file (FCN Segmentation output.)
        bin_mask = np.loadtxt(bin_mask_path, dtype=int)
        
        # Problem? We need to have similar pyramid downsacling that we are using for images, for binary mask input too.
        # The pyramid_gaussian function does not accept 2D list as input (only image). 
        # Solution: 
        # a- convert bin_mask to an image
        #bin_mask_img, colors = give_color_to_seg_img(bin_mask,n_classes)
        # b- feed it into pyramid_gaussian function
        #pyramid_bin_mask_img = tuple(pyramid_gaussian(bin_mask_img, downscale=1.5))
        
        # c- convert image into 2D list of int and save it back to bin_mask
        #pyramid_bin_mask = [get_seg_from_img(p) for p in pyramid_bin_mask_img]
        
        
        # task: get the threshold of the image
        
        crater_list_cnn = []
        
        winS = param.dmin
        # loop over the image pyramid
        
        for (i, resized) in enumerate(pyramid_gaussian(img, downscale=1.5)):
            if resized.shape[0] < 31:
                break
            
            print("Resized shape: %d, Window size: %d, i: %d" % (resized.shape[0], winS, i))
            #windowSize=(winS, winS)
            # loop over the sliding window for each layer of the pyramid
            # this process takes about 7 hours. To do quick test, we may try stepSize
            # to be large (60) and see if code runs OK
            idx = 0 
            for (x, y, window) in sliding_window(resized, stepSize=8, windowSize=(winS, winS)):
                
                # calcualte the number of 1 pixels in the binary mask area of the window.
                answ, crater_score = is_potential_crater(bin_mask,x,y,resized.shape[0], windowSize=(winS, winS)) 
                if answ :
                    # apply a circular mask ot the window here. Think about where you should apply this mask. Before, resizing or after it. 
                    crop_img =cv.resize(window, (50, 50))
                    cv.normalize(crop_img, crop_img, 0, 255, cv.NORM_MINMAX)
                    crop_img = crop_img.flatten()
                    
                    p_non, p_crater = cnn.predict([crop_img])[0]
                    
                    scale_factor = 1.5 ** i
                    x_c = int((x + 0.5 * winS) * scale_factor)
                    y_c = int((y + 0.5 * winS) * scale_factor)
                    crater_r = int(winS * scale_factor / 2)
                    
                    # add its probability to a score combined thresholded image and normal iamges. 
                    
                    if (p_crater + crater_score) / 2 >= 0.75:
                        crater_data = [x_c, y_c, crater_r  , p_crater]
                        crater_list_cnn.append(crater_data)
        
                   
        cnn_file = open("results/fcn-cnn/"+tile_img+"_sw_fcn_cnn.csv","w")
        with cnn_file:
            writer = csv.writer(cnn_file, delimiter=',')
            writer.writerows(crater_list_cnn)
        cnn_file.close()
        
        print("CNN detected ", len(crater_list_cnn), "craters for tile" + gt_num)
        print("The results is saved on results/fcn-cnn/"+tile_img+"_sw_fcn_cnn.csv file.")

print("end of main")
