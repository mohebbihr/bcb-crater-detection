#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 17:13:34 2018

@author: mohebbi
"""

import os
import pandas as pd
import cv2 as cv
import numpy
import Param
import random
from helper import sliding_window
import imutils

# This script will extract positive slides of the images. The size of each slice is 224 , 224
# each slice has 24 pixels over lap with each other. Therefore, stepSize = 224 - 24 = 200 
def extract_slices(gt_num, gt_img, savepath, windowSize=(224, 224), stepSize = 200, counter = 0):

    clone = gt_img.copy()
    # task: think about adding the pyramid in here.
    # or maybe generating slices at different size of gt images. From small to very big sizes and extract
    # with same size always. 

    for (x, y, window) in sliding_window(clone, stepSize, windowSize):

        dst_filename = os.path.join(savepath, "SL_"+ gt_num + "_" + str(counter) +  "_x_" + str(x) +"_y_" + str(y) +  ".jpg")
        cv.imwrite(dst_filename, window)
        counter +=1

    return counter

# This script extract TP and FN samples from tiles images and write it to approperiate directories.
# this function does not resize or normalize the extracted images. We do these parts on preprocess function in crater_preprocessing.py file. 

def extract_positive_samples(gt_num, gt_img, gt_data, gt_tp_savepath, param):

    clone = gt_img.copy()
    mask = numpy.zeros((gt_img.shape[0], gt_img.shape[1]))
    counter = 0

    x_gt_data = gt_data[0].values.tolist()
    y_gt_data = gt_data[1].values.tolist()
    d_gt_data = gt_data[2].values.tolist() # the third column is diameter

    for v in range(0,len(gt_data)):

        x_gt = int(round(x_gt_data[v]))
        y_gt = int(round(y_gt_data[v]))
        r_gt = int(round(d_gt_data[v] / 2))

        x = x_gt - r_gt
        y = y_gt - r_gt

        if x >=0 and y >=0 and x_gt + r_gt +1 <= gt_img.shape[0] and y_gt + r_gt +1 <= gt_img.shape[1] :

            crop_img = clone[y:y_gt + r_gt +1, x:x_gt + r_gt +1]

            # save rotations of the image too
            for degree in range(30,360,30):
                rotated = imutils.rotate_bound(crop_img, degree)
                dst_filename = os.path.join(gt_tp_savepath, "TP_"+ gt_num + "_" + str(counter) + "_" + str(degree) + ".png")
                cv.imwrite(dst_filename, rotated)
                counter += 1

            dst_filename = os.path.join(gt_tp_savepath, "TP_"+ gt_num + "_" + str(counter) + ".png")
            cv.imwrite(dst_filename, crop_img)
            # do the same for the mask too
            mask[y:y_gt + r_gt + 1, x:x_gt + r_gt + 1] = 1
            counter += 1

    return counter, mask

def extract_negative_samples(gt_num,gt_img, gt_mask, num_positive_samples, gt_fn_savepath, param, circleMask = False):

    clone = gt_img.copy()
    nn = 0
    while nn < num_positive_samples:

        aux_r = random.randint(param.dmin, param.dmax)/2
        aux_x = random.randint(0,gt_img.shape[0])
        aux_y = random.randint(0,gt_img.shape[1])

        s_x = aux_x - aux_r
        s_y = aux_y - aux_r

        e_x = aux_x + aux_r + 1
        e_y = aux_y + aux_r + 1

        if s_x >= 0 and s_y >=0 and e_x <= gt_img.shape[0] and e_y <= gt_img.shape[1]:

            # calculate its intersection with gt
            mask = numpy.zeros((gt_img.shape[0], gt_img.shape[1]))

            mask[s_y:e_y,s_x: e_x] = 1
            # element wise matrix multiplication
            threshold = numpy.sum(numpy.multiply(mask, gt_mask)) / numpy.sum(mask)

            if threshold <= param.thresh_overlay:

                crop_img = clone[s_y:e_y, s_x:e_x]
                dst_filename = os.path.join(gt_fn_savepath, "FN_"+ gt_num + "_" + str(nn) + ".png")
                if circleMask:
                        crop_circle_mask = numpy.zeros(crop_img.shape, dtype='uint8')
                        cv.circle(crop_circle_mask, (aux_x, aux_y), aux_r, (255, 255, 255), -1)
                        masked_crop = numpy.bitwise_and(crop_img, crop_circle_mask)
                        cv.imwrite(dst_filename, masked_crop)
                else:
                        cv.imwrite(dst_filename, crop_img)

                nn += 1

    return nn
    

if __name__ == "__main__":
    
    param = Param.Param()
    gt_list = ["1_24", "1_25", "2_24", "2_25", "3_24", "3_25"]

    for gt_num in gt_list:
        
        org_tp_savepath = os.path.join("crater_data", "samples", "org", "tile" + gt_num, "crater")
        org_fn_savepath = os.path.join("crater_data","samples", "org", "tile" + gt_num, "non-crater")
        th_org_tp_savepath = os.path.join("crater_data", "samples", "th_org", "tile" + gt_num, "crater")
        th_org_fn_savepath = os.path.join("crater_data","samples", "th_org", "tile" + gt_num, "non-crater")
        mask_tp_savepath = os.path.join("crater_data", "samples", "mask", "tile" + gt_num, "crater")
        mask_fn_savepath = os.path.join("crater_data","samples", "mask", "tile" + gt_num, "non-crater")
        th_mask_tp_savepath = os.path.join("crater_data", "samples", "th_mask", "tile" + gt_num, "crater")
        th_mask_fn_savepath = os.path.join("crater_data","samples", "th_mask", "tile" + gt_num, "non-crater")
        
        gt_csv_path = os.path.join("crater_data","gt", gt_num + "_gt.csv")
        gt_mask_img_path = os.path.join("crater_data","masks", "tile" + gt_num + ".pgm") # we use the mask of original image to extract true positive samples.
        gt_th_mask_img_path = os.path.join("crater_data","masks", "tile" + gt_num + "_th.pgm") # we add threshold image too. 
        gt_img_path = os.path.join("crater_data","tiles", "tile" + gt_num + ".pgm") # we use original image to extract false negative samples.
        gt_img_path_reflected_pgm = os.path.join("crater_data","tiles", "tile" + gt_num + "_reflected.pgm")
        gt_img_path_reflected_png = os.path.join("crater_data","tiles", "tile" + gt_num + "_reflected.png")
        gt_mask_img_path_reflected_pgm = os.path.join("crater_data","masks", "tile" + gt_num + "_reflected.pgm")
        gt_mask_img_path_reflected_png = os.path.join("crater_data","masks", "tile" + gt_num + "_reflected.png")
        gt_th_img_path = os.path.join("crater_data","th-images", "tile" + gt_num + ".pgm")
        #slices addresses. We store wit 24pixels overlap (_24overlap) and no overlap (_noverlap)
        gt_slices_path = os.path.join("crater_data", "slices","org_24overlap" , "tile" + gt_num)
        gt_mask_slices_path = os.path.join("crater_data", "slices","mask_24overlap","mask" , "tile" + gt_num)
        gt_slices_path2 = os.path.join("crater_data", "slices","org_noverlap" , "tile" + gt_num)
        gt_mask_slices_path2 = os.path.join("crater_data", "slices","mask_noverlap" , "tile" + gt_num)
        
        print("_____extracting positive and negative samples_______")
        
        gt_data = pd.read_csv(gt_csv_path, header=None)
        gt_img = cv.imread(gt_img_path)
        gt_th_img = cv.imread(gt_th_img_path)
        gt_mask_img = cv.imread(gt_mask_img_path)
        gt_th_mask_img = cv.imread(gt_th_mask_img_path)
        
        # extract positive and negative samples from org images and save it on org folder on samples directory.
        num_tp_samples, gt_mask = extract_positive_samples(gt_num, gt_img, gt_data, org_tp_savepath, param) # original photo 
        print(str(num_tp_samples) + " crater samples are extracted from " + gt_num + " tile.")
        num_fn_samples = extract_negative_samples(gt_num,gt_img,gt_mask, num_tp_samples, org_fn_savepath, param)
        print(str(num_fn_samples) + " non-crater samples are extracted from " + gt_num + " tile.")

        # extract positive and negative samples from threshold org images and save it on th_org folder on samples directory.
        num_tp_samples, gt_mask = extract_positive_samples(gt_num, gt_th_img , gt_data, th_org_tp_savepath, param) # threshold of the original photo 
        print(str(num_tp_samples) + " crater samples are extracted from threshold " + gt_num + " tile.")
        num_fn_samples = extract_negative_samples(gt_num,gt_th_img,gt_mask, num_tp_samples, th_org_fn_savepath, param)
        print(str(num_fn_samples) + " non-crater samples are extracted from threshold " + gt_num + " tile.")

        # extract positive and negative samples from threshold org images and save it on th_org folder on samples directory.
        num_tp_samples, gt_mask = extract_positive_samples(gt_num, gt_mask_img , gt_data, mask_tp_savepath, param) # mask of the original photo 
        print(str(num_tp_samples) + " crater samples are extracted from mask of  " + gt_num + " tile.")
        num_fn_samples = extract_negative_samples(gt_num,gt_img,gt_mask, num_tp_samples, mask_fn_savepath, param, True) # the non-crater areas of the masked image are black and therefore we need to pass the original image and then apply mask for each sample. 
        print(str(num_fn_samples) + " non-crater samples are extracted from mask of " + gt_num + " tile.")

        # extract positive and negative samples from threshold org images and save it on th_org folder on samples directory.
        num_tp_samples, gt_mask = extract_positive_samples(gt_num, gt_th_mask_img , gt_data, th_mask_tp_savepath, param) # original photo 
        print(str(num_tp_samples) + " crater samples are extracted from threshold mask of " + gt_num + " tile.")
        num_fn_samples = extract_negative_samples(gt_num,gt_th_img,gt_mask, num_tp_samples, th_mask_fn_savepath, param, True)
        print(str(num_fn_samples) + " non-crater samples are extracted from threshold " + gt_num + " tile.")
        
        print("______exctracting image slices (224,224)_______")
        print("______exctracting image slices without padding_______")
        
        num_slices = extract_slices(gt_num, gt_img, gt_slices_path, windowSize=(224, 224), stepSize = 200)
        num_slices += extract_slices(gt_num, gt_mask_img, gt_mask_slices_path, windowSize=(224, 224), stepSize = 200)
        
        # we don't need to extract slices without overlap from the original image. Because we only use slices from the 
        # reflected (with padding) image for testing purposes. 
        #extract slices without overlap too stepsize = 224
        #num_slices += extract_slices(gt_num, gt_img, gt_slices_path2, windowSize=(224, 224), stepSize = 224)
        #num_slices += extract_slices(gt_num, gt_mask_img, gt_mask_slices_path2, windowSize=(224, 224), stepSize = 224)
        
        print("______save tile images with padding, new size: 1792 x 1792_______")
        #gt_img_resized = cv.resize(gt_img, (1792, 1792))
        #gt_mask_img_resized = cv.resize(gt_mask_img, (1792, 1792))
        gt_img_reflected = cv.copyMakeBorder(gt_img,46,46,46,46,cv.BORDER_REFLECT)
        gt_mask_img_reflected = cv.copyMakeBorder(gt_mask_img,46,46,46,46,cv.BORDER_REFLECT)
        cv.imwrite(gt_img_path_reflected_pgm,gt_img_reflected)
        cv.imwrite(gt_img_path_reflected_png,gt_img_reflected)
        cv.imwrite(gt_mask_img_path_reflected_pgm,gt_mask_img_reflected)
        cv.imwrite(gt_mask_img_path_reflected_png,gt_mask_img_reflected)
        
        print("______exctracting image slices with padding with no overlap between slices_______")
        #extract slices without overlap too stepsize = 224
        num_slices += extract_slices(gt_num, gt_img_reflected, gt_slices_path2, windowSize=(224, 224), stepSize = 224)
        num_slices += extract_slices(gt_num, gt_mask_img_reflected, gt_mask_slices_path2, windowSize=(224, 224), stepSize = 224)
        
        print(str(num_slices) + " slices are extracted from " + gt_num + " tile.")
        
        
        