from skimage.transform import pyramid_gaussian
import cv2 as cv
from helper import sliding_window
import time
import os
import csv
from crater_cnn import Network
from crater_plots import plot_image, plot_conv_weights, plot_conv_layer

# running sliding window on each tile. This code has low accuracy and it is old. slow...


cwd = os.getcwd()

model = Network(img_shape=(30, 30, 1))
model.add_convolutional_layer(5, 16)
model.add_convolutional_layer(5, 36)
model.add_flat_layer()
model.add_fc_layer(size=128, use_relu=True)
model.add_fc_layer(size=2, use_relu=False)
model.finish_setup()
# model.set_data(data)

model_path = os.path.join(cwd, 'model.ckpt')
model.restore(model_path)
print("Model is loaded.")

# go through all the tile folders
gt_list = ["1_24", "1_25", "2_24", "2_25", "3_24", "3_25"]
path = 'crater_data/tiles/' # input path
outputpath = 'results/cnn/'

for gt_num in gt_list:
    tilefn = 'tile' + gt_num
    img = cv.imread(path + tilefn +'.pgm', 0)
    img = cv.normalize(img, img, 0, 255, cv.NORM_MINMAX)/255.0
    
    crater_list = []

    win_sizes = range(20, 30, 2)
    # loop over the image pyramid
    for (i, resized) in enumerate(pyramid_gaussian(img, downscale=1.5)):
        if resized.shape[0] < 30:
            break
        for winS in win_sizes:
            # loop over the sliding window for each layer of the pyramid
            for (x, y, window) in sliding_window(resized, stepSize=60, windowSize=(winS, winS)):
                # since we do not have a classifier, we'll just draw the window
                clone = resized.copy()
                y_b = y + winS
                x_r = x + winS
                crop_img = clone[y:y_b, x:x_r]
                crop_img =cv.resize(crop_img, (30, 30))
                crop_img = crop_img.flatten()
                p_non, p_crater = model.predict([crop_img])[0]
                scale_factor = 1.5 ** i
                if p_crater >= 0.5:
                    x_c = int((x + 0.5 * winS) * scale_factor)
                    y_c = int((y + 0.5 * winS) * scale_factor)
                    crater_size = int(winS * scale_factor)
                    crater_data = [x_c, y_c, crater_size, p_crater, 1]
                    crater_list.append(crater_data)
                # if we want to see where is processed.
                # cv.rectangle(clone, (x, y), (x + winS, y + winS), (0, 255, 0), 2)
                # cv.imshow("Window", clone)
                # cv.waitKey(1)
    out = csv.writer(open(tilefn + "_craters.csv","w"), delimiter=',',quoting=csv.QUOTE_ALL)
    out.writerow(crater_list)
    
    