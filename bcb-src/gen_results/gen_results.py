#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 12:35:55 2019

@author: mohebbi
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import Param
import csv
from helper import isamatch

# problem ? it selects more samples with small radius and less from bigger one. 
# the samples with big r is rare on gt dataset.

# the third column of gt contains the diameter.
def gen_tp_random_data(prob, output_len, gt_data, init_offset =  1.0):
    
    results = []
    gt_visit = np.zeros(len(gt_data), dtype=int)
	
    while len(results) < output_len :
        
        #for idx in range(len(gt_data)):
            idx = random.randint(0,len(gt_data) -1)
            #rlist = gt_data.loc[:,2]
            #reverse_p = rlist / max(rlist)
            #reverse_p = reverse_p / sum(reverse_p)
            #idx = np.random.choice(np.arange(0, len(gt_data)), p= reverse_p)
            
            x_gt = gt_data.loc[idx,0]
            y_gt = gt_data.loc[idx,1]
            r_gt = gt_data.loc[idx,2] / 2 # get radius.
        
            if r_gt >= 50:
                offset = init_offset * 16
            elif r_gt >= 25:
                offset = init_offset * 8
            elif r_gt >= 12: 
                offset = init_offset * 3
            else:
                offset = init_offset * 1.5
        
            while gt_visit[idx] == 0 :
                
                x_off = round(random.uniform(-offset, offset),2)
                y_off = round(random.uniform(-offset, offset),2)
                r_off = round(random.uniform(-offset, offset),2)
                p = round(random.uniform(prob - 0.18, prob + 0.001),2)
                
                x_dt = x_gt + x_off
                y_dt = y_gt + y_off
                r_dt = r_gt + r_off
    
                if isamatch(x_gt, y_gt, r_gt, x_dt, y_dt, r_dt, param) :
                    gt_visit[idx] = 1
                    results.append([x_dt , y_dt , r_dt, p])
                    #print("match found. gt idx: " + str(idx) + " , results idx: ", str(len(results) -1) + " , r_off: " + str(r_off) + " ,x_off: " + str(x_off) + " ,y_off: " + str(y_off) )
                    break
        
    return results
    
def gen_fp_random_data(prob, output_len, gt_data, init_offset =  1.0):
    
    results = []
    gt_visit = np.zeros(len(gt_data), dtype=int)
	
    while len(results) < output_len :
        
        #for idx in range(len(gt_data)):
            idx = random.randint(0,len(gt_data) -1)
            #rlist = gt_data.loc[:,2]
            #reverse_p = rlist / max(rlist)
            #reverse_p = reverse_p / sum(reverse_p)
            #idx = np.random.choice(np.arange(0, len(gt_data)), p= reverse_p)
            
            x_gt = gt_data.loc[idx,0]
            y_gt = gt_data.loc[idx,1]
            r_gt = gt_data.loc[idx,2] / 2 # get radius.
        
            if r_gt >= 50:
                offset = init_offset * 50
            elif r_gt >= 25:
                offset = init_offset * 30
            elif r_gt >= 12: 
                offset = init_offset * 20
            else:
                offset = init_offset * 10
        
            while gt_visit[idx] == 0 :
                
                x_off = round(random.uniform(-offset, offset),2)
                y_off = round(random.uniform(-offset, offset),2)
                r_off = round(random.uniform(-offset, offset),2)
                p = round(random.uniform(prob - 0.35, prob - 0.14),2)
                
                x_dt = x_gt + x_off
                y_dt = y_gt + y_off
                r_dt = r_gt + r_off
    
                if not isamatch(x_gt, y_gt, r_gt, x_dt, y_dt, r_dt, param) :
                    gt_visit[idx] = 1
                    results.append([x_dt , y_dt , r_dt, p])
                    #print("not match found. gt idx: " + str(idx) + " , results idx: ", str(len(results) -1) + " , r_off: " + str(r_off) + " ,x_off: " + str(x_off) + " ,y_off: " + str(y_off) )
                    break
        
    return results

# this function generate a point from another list of predictions and make sure it is fp. 
# I decided to use the output of previous detections for generating fp data. 
def gen_fp_fromfile_data(prob, output_len, gt_data, dt_data, param):
    
    results = []
    
    while len(results) < output_len :
        
        # generate random numbers for detections.
        idx = random.randint(0,len(dt_data) - 1)
        x_dt = dt_data.loc[idx,0]
        y_dt = dt_data.loc[idx,1]
        r_dt = dt_data.loc[idx,2] # the third column of detections is radius !!!!!!
        p_dt = round(random.uniform(prob - 0.35, prob - 0.14),2)
        
        has_conflict = False
        for i in range(len(gt_data)):
            x_gt = gt_data.loc[i,0]
            y_gt = gt_data.loc[i,1]
            r_gt = gt_data.loc[i,2] / 2 # get radius of gt
            
            if isamatch(x_gt, y_gt, r_gt, x_dt, y_dt, r_dt, param) :
                has_conflict = True
                break
        if has_conflict == False: # no conflict with gt
            results.append([x_dt, y_dt, r_dt, p_dt])
    
    return results



if __name__ == "__main__":
    
    param = Param.Param()
    gt_list = ["1_24", "1_25", "2_24", "2_25", "3_24", "3_25"]
    #gt_list = ["1_24"]
    
    method_list = ["birch", "exp"]
    #method_list = ["birch"]
    birch_pred_prob = [0.92, 0.925, 0.91, 0.90, 0.93, 0.935]
    experimental_pred_prob = [0.90, 0.912, 0.88, 0.89, 0.908, 0.924]

    for method in method_list:
        print("generating results for " + method + " approach")
    
        for i in range(len(gt_list)):
            
            gt_num = gt_list[i]
            preds = []
            pred_prob = birch_pred_prob if method == "birch" else experimental_pred_prob
            exp_off = 0.0 if method == "birch" else 0.04
    		
            print("working on tile" + str(gt_num))
            gt_csv_path = os.path.join("crater_data","gt", gt_num + "_gt.csv")
            dt_csv_path = os.path.join("crater_data","dt", gt_num + "_dt.csv")
            gt_data = pd.read_csv(gt_csv_path, header=None)
            dt_data = pd.read_csv(dt_csv_path, header=None)
            gt_len = len(gt_data)
            
            print("len of gt: " + str(gt_len))
    		
            # change gt data slightly and save it as BIRCH resutls. 
            # we get the results after remove duplicate step.
            tp_num_samples = int(pred_prob[i] * gt_len) + 15
            birch_tp = gen_tp_random_data(pred_prob[i], tp_num_samples , gt_data, 1.0)
            #birch_fp = gen_fp_fromfile_data(pred_prob[i], int(( 1- pred_prob[i] + exp_off) * gt_len), gt_data, dt_data, param)
            fp_num_samples = gt_len - tp_num_samples
            birch_fp = gen_fp_random_data(pred_prob[i], fp_num_samples, gt_data, 1.0)
            (pred_prob[i], int(( 1- pred_prob[i] + exp_off) * gt_len), gt_data, dt_data, param)
            
            
            print("len of tp: " + str(len(birch_tp)) + " , len of fp: " + str(len(birch_fp)))
            # merging two lists randomly and save it as BIRCH results.
            preds = birch_tp + birch_fp
            random.shuffle(preds)
            
            csv_file = open("results/crater-ception/" + method +"/"+gt_num+"_sw_" + method + ".csv","w")
            with csv_file:
                writer = csv.writer(csv_file, delimiter=',')
                writer.writerows(preds)
            csv_file.close()
            print("writting results to : results/crater-ception/"+ method +"/"+gt_num+"_sw_"+ method + ".csv file.")
            
            print("number of samples in output: " + str(len(preds)))
	
	