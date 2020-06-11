import cv2 as cv
import time
from datetime import timedelta
import os
import pandas as pd
from helper import calculateDistance, BIRCH_duplicate_removal,BIRCH2_duplicate_removal, Banderia_duplicate_removal, XMeans_duplicate_removal, draw_craters_rectangles, draw_craters_circles, evaluate
from non_max_suppression import NMS
import Param

# the raw data to process for duplicate removal
param = Param.Param()
removal_method = 'NMS'
csv_path = 'results/cnn/tile1_24_sw_cnn.csv'
gt_csv_path = 'crater_data/gt/1_24_gt.csv'
save_path = 'results/cnn/evaluations/' + removal_method + '/tile1_24_cnn'
testset_name = 'tile1_24'

# the image for drawing rectangles
img_path = os.path.join('crater_data', 'images', testset_name + '.pgm')
gt_img = cv.imread(img_path)

data = pd.read_csv(csv_path, header=None)
gt = pd.read_csv(gt_csv_path, header=None)

threshold = 0.75

start_time = time.time()

# first pass, remove duplicates for points of same window size
df1 = {}
for ws in data[2].unique():
    if (ws >= param.dmin) and (ws <= param.dmax):
        df1[ws] = data[ (data[3] > 0.75) & (data[2] == ws) ] # take only 75% or higher confidence
        df1[ws] = NMS(df1[ws])

# Start merging process
# We will add points of greatest size first
# then merge with the next smaller size and remove duplicates
# Do this until the smallest window size has been included

merge = pd.DataFrame()
for ws in reversed(sorted(df1.keys())):
    merge = pd.concat([merge, df1[ws]])
    old_size = len(merge)
    #merge = BIRCH2_duplicate_removal(merge, threshold) # we can tweak ws for eliminations
    merge = NMS(merge) 
    new_size = len(merge)
    print("Processed window size", ws, ", considered", old_size, "points, returned", new_size, "points")

# save the no duplicate csv file
merge[[0,1,2]].to_csv("%s_noduplicates.csv" % save_path, header=False, index=False)
craters = merge[[0,1,2]]

# evaluate with gt and draw it on final image.
dr, fr, qr, bf, f_measure, tp, fp, fn  = evaluate(craters, gt, gt_img, 64, True, save_path, param)


end_time = time.time()
time_dif = end_time - start_time
print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))