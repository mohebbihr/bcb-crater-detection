#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 17:10:35 2018

@author: mohebbi
"""
from helper import isamatch
import numpy as np
import cv2 as cv
import os

# save_path: the general path to save the extracted samples 
# img_type: crater or non-crater
#
def save_samples(info_list, img, save_path, img_type, img_dimensions=(50, 50)):
    
    clone = img.copy()
    counter = 1
    for (x_c,y_c,r) in info_list:
        x = x_c - r
        y = y_c - r
        
        crop_img = clone[y:y_c + r, x:x_c + r]
        scaled_img =cv.resize(crop_img, img_dimensions)
        
        cv.normalize(scaled_img, scaled_img, 0, 255, cv.NORM_MINMAX)
		dst_filename = os.path.join(save_path, img_type, str(counter) + "_hng.jpg")
        cv.imwrite(dst_filename, scaled_img)
		counter += 1
        
	return counter -1

def extract_hard_negative_samples(craters, gt, img, save_path, img_dimensions, param):
    #sort by radius
    gt = gt.sort_values(by=[2]).values
    dt = craters.sort_values(by=[2]).values
    
    gt_visit = np.zeros(len(gt), dtype=int)
    dt_visit = np.zeros(len(dt), dtype=int)
    
    for v in range(0,len(gt)):
        x_gt = gt[v][0]
        y_gt = gt[v][1]
        r_gt = gt[v][2]
        
        for w in range(0,len(dt)):
            x_dt = dt[w][0]
            y_dt = dt[w][1]
            r_dt = dt[w][2]
            
            if( gt_visit[v] == 0  and isamatch(x_gt, y_gt, r_gt, x_dt, y_dt, r_dt, param)):
                
                gt_visit[v] = 1
                dt_visit[w] = 1

    # indexes that we missed from gt (crater)
    fn_index = [i for i, e in enumerate(gt_visit) if e == 0]
    # wrong predictions (non-crater)
    fp_index = [i for i, e in enumerate(dt_visit) if e == 0]
    
    # extract the x,y and r of indexs
    fn_info = []
    fp_info = []
    
    for i in fn_index:
        fn_info.append([gt[i][0], gt[i][1], gt[i][2]])
        
    for i in fp_index:
        fp_info.append([dt[i][0], dt[i][1], dt[i][2]])
        
    # extract locations from image and save it on save_path
    num_fn_samples = save_samples(fn_info, img, save_path, "crater", img_dimensions)
    num_fp_samples = save_samples(fp_info, img, save_path, "non-crater", img_dimensions)
	
    print( str(num_fn_samples) + " crater hard negative samples are extracted.")
    print( str(num_fp_samples) + " non-crater hard negative samples are extracted.")
    
    
    