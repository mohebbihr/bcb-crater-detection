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
import numpy as np

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

x = merge[0].values.tolist()
y = merge[1].values.tolist()

X = np.column_stack((x, y))

plt.scatter(X[:, 0], X[:, 1], s=50);
plt.savefig(save_path +'_org.png', bbox_inches='tight', dpi=400)

kmeans = KMeans(n_clusters=8)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
plt.savefig(save_path +'_cluster.png', bbox_inches='tight', dpi=400)


end_time = time.time()
time_dif = end_time - start_time
print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))