import cv2 as cv
import time
from datetime import timedelta
import os
import pandas as pd
from helper import calculateDistance, BIRCH_duplicate_removal, Banderia_duplicate_removal, XMeans_duplicate_removal, draw_craters_rectangles, draw_craters_circles, evaluate_cmp, evaluate
import Param

# the raw data to process for duplicate removal
param = Param.Param()
removal_method1 = 'NMS'
removal_method2 = 'BIRCH'

csv_path = 'results/cnn/west_train_west_test_1_24_cnn.csv'
gt_csv_path = 'crater_data/gt/gt_tile1_24.csv'

path1 = 'results/cnn/evaluations/' + removal_method1 + '/west_train_west_test_1_24_cnn_noduplicates.csv'
path2 = 'results/cnn/evaluations/' + removal_method2 + '/west_train_west_test_1_24_cnn_noduplicates.csv'

save_path = 'results/cnn/evaluations/NMS_BIRCH/west_train_west_test_1_24_cnn'
testset_name = 'tile1_24'

# the image for drawing rectangles
img_path = os.path.join('crater_data', 'images', testset_name + '.pgm')
gt_img = cv.imread(img_path)


gt = pd.read_csv(gt_csv_path, header=None)

no_dup_data1 = pd.read_csv(path1, header=None)
no_dup_data2 = pd.read_csv(path2, header=None)

start_time = time.time()

# compare the results of two duplicate removal methods
#evaluate_cmp(no_dup_data1, no_dup_data2, gt, gt_img, 64, True, save_path, param)
evaluate(gt, gt, gt_img, 64, True, save_path, param)
#img = draw_craters_rectangles(img_path, merge, show_probs=False)
#img = draw_craters_circles(img_path, merge, show_probs=False)
#cv.imwrite("%s.jpg" % (csv_path.split('.')[0]), img, [int(cv.IMWRITE_JPEG_QUALITY), 100])


end_time = time.time()
time_dif = end_time - start_time
print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))