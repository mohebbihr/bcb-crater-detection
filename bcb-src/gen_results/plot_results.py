import time
from datetime import timedelta
import os
import pandas as pd
from helper import evaluate
import Param
import cv2 as cv

start_time = time.time()
# the data after duplicate removal
param = Param.Param()

#method_list = ["birch", "exp"]
method_list = ["birch"]
gt_list = ["1_24", "1_25", "2_24", "2_25", "3_24", "3_25"]
#gt_list = ["1_24"]

for method in method_list:
    print("evaluation of " + method + " approach")
    for gt in gt_list:
        # the image for drawing rectangles
        print("working on tile" + gt)
        tile_name = "tile" + gt
        img_path = os.path.join('crater_data', 'tiles', tile_name + '.pgm')
        gt_img = cv.imread(img_path)
        gt_csv_path = os.path.join('crater_data', 'gt', gt + '_gt.csv')
        gt_data = pd.read_csv(gt_csv_path, header=None)
		
        # read detection from csv file.
        dt_csv_path = os.path.join('results', 'crater-ception', method, gt + '_sw_' + method + '.csv')
        craters = pd.read_csv(dt_csv_path, header=None)
        print("reading from file: " + str(dt_csv_path))
        
        # save results path
        save_path = 'results/crater-ception/' + method + '/evaluations/' + gt + '_sw_' + method
        craters = craters[[0,1,2]]
        # evaluate with gt and draw it on final image.
        evaluate(craters, gt_data, gt_img, 64, True, save_path, param)

end_time = time.time()
time_dif = end_time - start_time
print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))