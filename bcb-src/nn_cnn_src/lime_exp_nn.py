#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 12:40:45 2018

@author: mohebbi
"""

import numpy as np
import cv2

import os
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

from crater_cnn import Network
from crater_plots import plot_image, plot_conv_weights, plot_conv_layer
from keras.applications import imagenet_utils
from keras.preprocessing import image
from skimage.color import gray2rgb, rgb2gray
cwd = os.getcwd()

input_shape = (50, 50)
preprocess = imagenet_utils.preprocess_input
# setup NN
nn = Network(img_shape=(50, 50, 1))
nn.add_flat_layer()
nn.add_fc_layer(size=50 * 50, use_relu=True)
nn.add_fc_layer(size=16, use_relu=True)
nn.add_fc_layer(size=2, use_relu=False)
nn.finish_setup()
# model.set_data(data)

# restore previously trained CNN model
print("loading the pre-trained NN model")
nn_model_path = os.path.join(cwd, 'results', 'nn_models', 'crater_east_model_nn.ckpt')
nn.restore(nn_model_path)

def transform_img_fn(path_list):
    org_out = []
    trans_out = []
    for img_path in path_list:
        src_img = cv2.imread(img_path)
        gray_img = rgb2gray(src_img)
        scaled_img = cv2.resize(gray_img, input_shape)
        norm_img = cv2.normalize(scaled_img, scaled_img, 0, 255, cv2.NORM_MINMAX)
        #flat_img = norm_img.flatten()
        
        #img = image.load_img(img_path, target_size=input_shape)
        #x = image.img_to_array(img)
        #x = np.expand_dims(x, axis=0)
        #x = preprocess(x)
        org_out.append(src_img)
        trans_out.append(norm_img)
    return org_out, trans_out

def predict_fn(images):
    out = []
    for img in images:
        # make the img gray again!!
        img = rgb2gray(img)
        scaled_img = cv2.resize(img, input_shape)
        norm_img = cv2.normalize(scaled_img, scaled_img, 0, 255, cv2.NORM_MINMAX)
        flat_img = norm_img.flatten()
        p_non, p_crater = nn.predict([flat_img])[0]
        
        #if p_crater >= 0.5 :
        out.append(p_crater)
        #else:
        out.append(p_non)
    
    return out
    

# get some test image
images, trans_images = transform_img_fn(['./crater_data/images/tile1_24/crater/TE_tile1_24_001.jpg'])
print(images[0].shape)
print(trans_images[0].shape)

#print("showing image")
#plt.imshow(images[0], cmap='gray')
#plt.show()

# get prediction
print("predict class probabilities")
preds = predict_fn(images)
print(preds)


explainer  = lime_image.LimeImageExplainer()

explanation = explainer.explain_instance(images[0], predict_fn, hide_color=0, num_samples=1000)

temp, mask = explanation.get_image_and_mask(240, positive_only=True, num_features=5, hide_rest=True)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
plt.show()

