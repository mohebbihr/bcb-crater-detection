#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 16:15:33 2019

@author: mohebbi
"""
import os
import pandas as pd
import numpy as np
import seaborn as sns

if __name__ == "__main__":

    gt_list = ["1_24", "1_25", "2_24", "2_25", "3_24", "3_25"]
    #gt_list = ["1_24", "1_25"]

    acc_r = []
    
    for gt_num in gt_list:
        
        print(str(gt_num) + " elements added to r list")
        gt_csv_path = os.path.join("crater_data","gt", gt_num + "_gt.csv")
        gt_data = pd.read_csv(gt_csv_path, header=None)
        
        x = gt_data.loc[:,0]
        y = gt_data.loc[:,1]
        r = gt_data.loc[:,2]
        print(str(len(r)) + " elements added to r list")
        for v in r.get_values():
            acc_r.append(v)
        # we use the kernel-density plot of x and y
        #ax_x_y = sns.kdeplot(x, y)
        #ax_x_y.figure.savefig("tile" + gt_num + "_gt_x_y_kdeplot.png")
        
    # kernel-density plot of r
    ax_r = sns.kdeplot(r * 2, cut=0,bw=.25)
    ax_r.set_title("Kernel Density Plot of Ground Truth Image Sizes")
    ax_r.set_xlabel("Image Size")
    ax_r.set_ylabel("Kernel Density Estimate")
    ax_r.set_xlim(0,200)
    #ax_r.set_ylim(0,0.2)
    ax_r.figure.savefig("all_tiles_r_kdeplot.png", bbox_inches='tight', dpi=400)
    #ax_r.figure.savefig("tile" + gt_num + "_gt_r_kdeplot.png", bbox_inches='tight', dpi=400)
    print("Saving the kernel density plot of all samples in ground truth to disk is done.")    
        
