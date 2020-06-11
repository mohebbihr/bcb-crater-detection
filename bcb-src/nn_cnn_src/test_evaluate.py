import cv2 as cv
import time
from datetime import timedelta
import os
import pandas as pd
from helper import calculateDistance, BIRCH_duplicate_removal,BIRCH2_duplicate_removal, Banderia_duplicate_removal, XMeans_duplicate_removal, draw_craters_rectangles, draw_craters_circles, evaluate
import Param

# the raw data to process for duplicate removal
param = Param.Param()
#removal_method = 'BIRCH'
removal_method = 'Banderia'
testset_name = 'tile2_25'

gt_csv_path = 'crater_data/gt/gt_' + testset_name + '.csv'
csv_path = 'results/cnn/evaluations/' + removal_method + '/west_train_center_test_2_25_cnn_noduplicates.csv'
save_path = 'results/cnn/evaluations/' + removal_method + '/west_train_center_test_2_25_cnn'

# the image for drawing rectangles
img_path = os.path.join('crater_data', 'images', testset_name + '.pgm')
gt_img = cv.imread(img_path)

craters = pd.read_csv(csv_path, header=None)
gt = pd.read_csv(gt_csv_path, header=None)

start_time = time.time()



# evaluate with gt and draw it on final image.
dr, fr, qr, bf, f_measure, tp, fp, fn  = evaluate(craters, gt, gt_img, 64, True, save_path, param)

#img = draw_craters_rectangles(img_path, merge, show_probs=False)
#img = draw_craters_circles(img_path, merge, show_probs=False)
#cv.imwrite("%s.jpg" % (csv_path.split('.')[0]), img, [int(cv.IMWRITE_JPEG_QUALITY), 100])


end_time = time.time()
time_dif = end_time - start_time
print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))