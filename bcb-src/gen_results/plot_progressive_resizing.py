#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 12:35:55 2019

@author: mohebbi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    epochs = [i for i in range(0,525,25)]
    # validation accuracy of progressive models
    acc_1 = [0.44, 0.46, 0.48, 0.48, 0.5, 0.49, 0.51, 0.53, 0.56, 0.54, 0.55, 0.58, 0.58, 0.59, 0.58, 0.63, 0.64, 0.69, 0.70, 0.72, 0.72] # 12 x 12 images
    acc_2 = [0.42, 0.54, 0.56, 0.58, 0.57, 0.56, 0.56, 0.53, 0.59, 0.64, 0.65, 0.66, 0.65, 0.71, 0.74, 0.73, 0.77, 0.80, 0.86, 0.85, 0.846] # 24 x 24 images
    acc_3 = [0.51, 0.55, 0.56, 0.59, 0.60, 0.58, 0.61, 0.63, 0.62, 0.66, 0.68, 0.70, 0.71, 0.73, 0.74, 0.78, 0.79, 0.84, 0.85, 0.89, 0.892] # 48 x 48 images
    acc_list = [acc_1, acc_2, acc_3]
	
    progressive_resizing = ['12 x 12 ','24 x 24 ','48 x 48 ']
    color_sequence = ['#ff0000','#0000ff','#006400']
    fig, ax = plt.subplots(1, 1, figsize=(13, 9))
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_xlim(0, 500)
	
    plt.plot(epochs, acc_1, 'r--') 
    plt.plot(epochs, acc_2, 'bs')
    plt.plot(epochs, acc_2, 'b') 
    plt.plot(epochs, acc_3, 'g+') 
    plt.plot(epochs, acc_3, 'g')
	
    for i, step in enumerate(progressive_resizing):
        y_pos = acc_list[i][-1] - 0.005
        plt.text(505, y_pos, step, fontsize=14, color=color_sequence[i])
		
    #plt.show()
    plt.savefig('progressive_resizing_val_acc.png', bbox_inches='tight', dpi=400)
	
	