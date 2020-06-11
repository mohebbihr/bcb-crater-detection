from skimage.transform import pyramid_gaussian
import cv2 as cv
from helper import sliding_window
import os
import csv
from crater_cnn import Network as CNN
from crater_nn import Network as NN
import Param
import argparse

# This script will go through all image tiles and detects crater area using sliding window method.
# Then, write results as a csv file to the results folder. The results of this script is the input to the remove_duplicates.py script. 
# you need to provide the tile image name as argument after --tileimg command. For instance tile1_24

param = Param.Param()
cwd = os.getcwd()

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

for gt_num in gt_list:

    tile_img = 'tile' + gt_num
    
    path = os.path.join('crater_data', 'tiles')
    img = cv.imread(os.path.join(path, tile_img + '.pgm'), 0)
    img = cv.normalize(img, img, 0, 255, cv.NORM_MINMAX)/255.0
    
    # Task. Creat new script and Apply results of segmentation phase (FCN) to remove the non-crater areas of crater image.
    
    # task: get the threshold of the image
    
    crater_list_cnn = []
    #crater_list_nn = []
    
    winS = param.dmin
    # loop over the image pyramid
    
    for (i, resized) in enumerate(pyramid_gaussian(img, downscale=1.5)):
        if resized.shape[0] < 31:
            break
        
        print("Resized shape: %d, Window size: %d, i: %d" % (resized.shape[0], winS, i))
    
        # loop over the sliding window for each layer of the pyramid
        # this process takes about 7 hours. To do quick test, we may try stepSize
        # to be large (60) and see if code runs OK
        for (x, y, window) in sliding_window(resized, stepSize=8, windowSize=(winS, winS)):
            
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
            
            if p_crater >= 0.75:
                crater_data = [x_c, y_c, crater_r  , p_crater]
                crater_list_cnn.append(crater_data)
    
               
    cnn_file = open("results/cnn/"+tile_img+"_sw_cnn_th.csv","w")
    with cnn_file:
        writer = csv.writer(cnn_file, delimiter=',')
        writer.writerows(crater_list_cnn)
    cnn_file.close()
    
    print("CNN detected ", len(crater_list_cnn), "craters")
    print("The results is saved on results/cnn/"+tile_img+"_sw_cnn_th.csv file.")

