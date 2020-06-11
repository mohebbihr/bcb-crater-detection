import cv2 as cv
from helper import sliding_window
import time
import os
import csv
from crater_cnn import Network 
import pickle
import Param
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-tileimg", "--tileimg", type=str, default="tile3_25", help="The name of tile image")
args = vars(ap.parse_args())

param = Param.Param()
cwd = os.getcwd()

# setup NN
nn = Network(img_shape=(50, 50, 1))
nn.add_flat_layer()
nn.add_fc_layer(size=50 * 50, use_relu=True)
nn.add_fc_layer(size=16, use_relu=True)
nn.add_fc_layer(size=2, use_relu=False)
nn.finish_setup()
# model.set_data(data)

# restore previously trained CNN model
nn_model_path = os.path.join(cwd, 'models/nn/crater_model_nn.ckpt')
nn.restore(nn_model_path)

tile_img = args["tileimg"]

path = os.path.join('crater_data', 'images')
img = cv.imread(os.path.join(path, tile_img +'.pgm'), 0)
img = cv.normalize(img, img, 0, 255, cv.NORM_MINMAX)/255.0

crater_list_nn = []

win_sizes = range(param.dmin, param.dmax, 5)
# loop over the image pyramid

for winS in win_sizes:
    print("Resized shape: %d, Window size: %d" % (img.shape[0], winS))

    # loop over the sliding window for each layer of the pyramid
    # this process takes about 7 hours. To do quick test, we may try stepSize
    # to be large (60) and see if code runs OK
    #for (x, y, window) in sliding_window(resized, stepSize=2, windowSize=(winS, winS)):
    for (x, y, window) in sliding_window(img, stepSize=2, windowSize=(winS, winS)):
        # since we do not have a classifier, we'll just draw the window
        crop_img =cv.resize(window, (50, 50))
        crop_img = crop_img.flatten()
        
        p_non, p_crater = nn.predict([crop_img])[0]
        #nn_p = nn.feedforward_flat(crop_img)[0,0]
        
        x_c = (x + 0.5 * winS) 
        y_c = (y + 0.5 * winS) 
        crater_r = winS/2
        
        if p_crater >= 0.75:
            crater_data = [x_c, y_c, crater_r, p_crater]
            crater_list_nn.append(crater_data)

           
cnn_file = open("results/nn/"+tile_img+"_sw_nn.csv","w")
with cnn_file:
    writer = csv.writer(cnn_file, delimiter=',')
    writer.writerows(crater_list_nn)
cnn_file.close()

print("NN detected ", len(crater_list_nn), "craters")

