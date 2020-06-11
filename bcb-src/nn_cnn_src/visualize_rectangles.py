import cv2 as cv
import time
from datetime import timedelta
import os
import pandas as pd
cwd = os.getcwd()

sample_name = 'tile3_24'

path = os.path.join('crater_data', 'images')
gt_path = os.path.join('crater_data', 'gt', 'gt_%s.csv' % sample_name)
img = cv.imread(os.path.join(path, '%s.pgm' % sample_name), 1)
img = cv.normalize(img, img, 0, 255, cv.NORM_MINMAX)
img2 = img.copy()
img3 = img.copy()


cnn_data = pd.read_csv("crater_24_cnn.csv", names = ['x_c', 'y_c', 'crater_size', 'p_crater', 'label'])

#cnn_data[(cnn_data['p_crater']>0.99)&(cnn_data['crater_size']<50)].info()

nn_data = pd.read_csv("crater_24_nn.csv", names = ['x_c', 'y_c', 'crater_size', 'p_crater', 'label'])

gt_data = pd.read_csv(gt_path, names = ['x_c', 'y_c', 'crater_size'])

start_time = time.time()

cv.imwrite("%s_original.png" % sample_name, img)

for index, row in cnn_data[(cnn_data['p_crater'] > 0.5)&(cnn_data['crater_size'] < 25)].iterrows():
    winS = int(row['crater_size'])
    half_winS = int(winS/2)
    x = int(row['x_c'] - half_winS)
    y = int(row['y_c'] - half_winS)
    # if we want to see where is processed.
    cv.rectangle(img, (x, y), (x + winS, y + winS), (0, 255, 0), 2)
    
cv.imwrite("%s_cnn_detections.png" % sample_name, img)

for index, row in nn_data[(nn_data['p_crater'] > 0.5)&(nn_data['crater_size'] < 25)].iterrows():
    winS = int(row['crater_size'])
    half_winS = int(winS/2)
    x = int(row['x_c'] - half_winS)
    y = int(row['y_c'] - half_winS)
    # if we want to see where is processed.
    cv.rectangle(img2, (x, y), (x + winS, y + winS), (0, 255, 0), 2)
    
cv.imwrite("%s_nn_detections.png" % sample_name, img2)

for index, row in gt_data.iterrows():
    half_winS = int(row['crater_size'])
    winS = int(half_winS*2)
    x = int(row['x_c'] - half_winS)
    y = int(row['y_c'] - half_winS)
    # if we want to see where is processed.
    cv.rectangle(img3, (x, y), (x + winS, y + winS), (255, 0, 0), 2)
    
cv.imwrite("%s_gt.png" % sample_name, img3)

end_time = time.time()
time_dif = end_time - start_time
print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
