# this script determines the best K for kmeans using plotting (elbow method).
import cv2 as cv
import time
from datetime import timedelta
import os
import pandas as pd
from helper import calculateDistance, draw_craters_rectangles, draw_craters_circles, evaluate
import Param
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

# the raw data to process for duplicate removal
param = Param.Param()
removal_method = 'KMeans'
csv_path = 'results/cnn/west_train_west_test_1_25_cnn.csv'
save_path = 'results/cnn/evaluations/' + removal_method + '/west_train_west_test_1_25_cnn'
data = pd.read_csv(csv_path, header=None)

start_time = time.time()

# first pass, remove duplicates for points of same window size
df1 = {}
merge = pd.DataFrame()
for ws in data[2].unique():
    df1[ws] = data[ (data[3] > 0.75) & (data[2] == ws) ] # take only 75% or higher confidence
    merge = pd.concat([merge, df1[ws]])

distorsions = []
for k in range(2, 20):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(merge)
    distorsions.append(kmeans.inertia_)

fig = plt.figure(figsize=(15, 5))
plt.plot(range(2, 20), distorsions)
plt.grid(True)
plt.title('Elbow curve')
plt.savefig(save_path +'_elbow.png', bbox_inches='tight', dpi=400)


end_time = time.time()
time_dif = end_time - start_time
print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))