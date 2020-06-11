#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 17:13:34 2018

@author: mohebbi
"""
import sys
import os
import pandas as pd
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
#from helper import save_gt

# This script make the image mask of gt images. We extract training samples from mask images on the next step (extract_samples.py)

def save_gt(img,gt_data, save_path):
    nseg = 64    
    implot = plt.imshow(img.copy())
    x_gt_data = gt_data[0].values.tolist()
    y_gt_data = gt_data[1].values.tolist()
    d_gt_data = gt_data[2].values.tolist()
	
    for i in range(0, len(gt_data)):
        x = x_gt_data[i]
        y = y_gt_data[i]
        r = d_gt_data[i] / 2
        
        theta = np.linspace(0.0, (2 * math.pi), (nseg + 1))
        pline_x = np.add(x, np.dot(r, np.cos(theta)))
        pline_y = np.add(y, np.dot(r, np.sin(theta)))
        plt.plot(pline_x, pline_y, 'b-')
    
    
    plt.savefig(save_path +'_gt.png', bbox_inches='tight', dpi=400)
    plt.show()

def create_img_mask(gt_img, gt_data):
    
    clone = gt_img.copy()
    mask = np.zeros(gt_img.shape, dtype='uint8')
    nseg = 64    
    implot = plt.imshow(clone)
        
    x_gt_data = gt_data[0].values.tolist()
    y_gt_data = gt_data[1].values.tolist()
    d_gt_data = gt_data[2].values.tolist() # the third column is diameter
    
    for v in range(0,len(gt_data)):
        
        x_gt = int(round(x_gt_data[v]))
        y_gt = int(round(y_gt_data[v]))
        r_gt = int(round(d_gt_data[v] / 2))
        
        # create image mask
        cv.circle(mask, (x_gt, y_gt), r_gt, (255, 255, 255), -1)
        # use other method do draw cirles. When you zoon in you can see that it is not a complete circle and some non-crater areas are considered as crater area. 
        #theta = np.linspace(0.0, (2 * math.pi), (nseg + 1))
        #pline_x = np.add(x_gt, np.dot(r_gt, np.cos(theta)))
        #pline_y = np.add(y_gt, np.dot(r_gt, np.sin(theta)))
        #plt.plot(pline_x, pline_y, 'b-')
            	
    # apply the mask
    maskedImage = np.bitwise_and(clone, mask)	
	
    return maskedImage  


if __name__ == "__main__":
    
    img_dim = (50, 50)
    gt_list = ["1_24", "1_25", "2_24", "2_25", "3_24", "3_25"]

    for gt_num in gt_list:
        
        gt_tp_savepath = os.path.join("crater_data", "images", "tile" + gt_num, "crater")
        gt_fn_savepath = os.path.join("crater_data","images", "tile" + gt_num, "non-crater")
        gt_csv_path = os.path.join("crater_data","gt", gt_num + "_gt.csv")
        gt_img_path = os.path.join("crater_data","tiles", "tile" + gt_num + ".pgm")
        gt_th_img_path = os.path.join("crater_data","th-images", "tile" + gt_num + ".pgm")
        gt_mask_img_path = os.path.join("crater_data","masks", "tile" + gt_num + ".pgm")
        gt_th_mask_img_path = os.path.join("crater_data","masks", "tile" + gt_num + "_th.pgm")
        
        gt_data = pd.read_csv(gt_csv_path, header=None)
        gt_img = cv.imread(gt_img_path)
    
        # apply thresholding
        ret,th_img = cv.threshold(gt_img,127,255,cv.THRESH_BINARY)
        
        # save the thresholed image.
        cv.imwrite(gt_th_img_path, th_img)
        print(str(" The threshold image of " + gt_num + " tile is saved on th-images folder."))
    
		# save gt
        save_gt(gt_img, gt_data, "crater_data/tiles/tile" + gt_num)
        print(str(" The gt representation image of " + gt_num + " tile is saved on images folder."))
	
        gt_mask = create_img_mask(gt_img, gt_data)
        
        # save the mask image into masks folder
        cv.imwrite(gt_mask_img_path, gt_mask)
                
        print(str(" The mask of " + gt_num + " tile is saved on masks folder."))
        
        gt_th_mask = create_img_mask(th_img, gt_data)
        
        # save the mask image into masks folder
        cv.imwrite(gt_th_mask_img_path, gt_th_mask)
                
        print(str(" The therashold mask of " + gt_num + " tile is saved on masks folder."))
		
