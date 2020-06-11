#!/usr/bin/env python3

import csv
import os
from xmeans import XMeans
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

X_axis=[]
Y_axis=[]
W_size=[]
probs=[]
cwd = os.getcwd()
file=cwd +'/crater_25_cnn.csv'
#file=cwd +'/crater_24_sho_001.csv'
with open(file) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        if row:
            X_axis.append(float(row[0]))
            Y_axis.append(float(row[1]))
            W_size.append(float(row[2]))
            probs.append(float(row[3]))


xmax = np.max(X_axis)
ymax = np.max(Y_axis)
wmax = np.max(W_size)
X_axis=np.asarray(X_axis, dtype=np.float64)/xmax
Y_axis=np.asarray(Y_axis, dtype=np.float64)/ymax
W_size=np.asarray(W_size, dtype=np.float64)/wmax
a=np.c_[X_axis, Y_axis, W_size]

datafit = np.c_[X_axis, Y_axis, W_size]
kmeans = KMeans(n_clusters=214, max_iter=1000, tol=0.0001, algorithm='auto').fit(datafit)
x_means = XMeans(random_state=1).fit(np.c_[X_axis, Y_axis, W_size])

# print(x_means.labels_)
# print(x_means.cluster_centers_)
# print(x_means.cluster_log_likelihoods_)
# print(x_means.cluster_sizes_)

removed_list_cnn = []
for row in x_means.cluster_centers_:
    xc = row[0] * xmax
    yc = row[1] * ymax
    ws = row[2] * wmax
    removed_list_cnn.append([xc, yc, ws])
removed_file = open("crater_25_cnn_removed.csv","w", newline='')
with removed_file:
    writer = csv.writer(removed_file, delimiter=',')
    writer.writerows(removed_list_cnn)
removed_file.close()

kmeans_list_cnn = []
for row in kmeans.cluster_centers_:
    xc = row[0] * xmax
    yc = row[1] * ymax
    ws = row[2] * wmax
    kmeans_list_cnn.append([xc, yc, ws])
kmeans_file = open("crater_25_cnn_kmeans.csv","w", newline='')
with kmeans_file:
    writer = csv.writer(kmeans_file, delimiter=',')
    writer.writerows(kmeans_list_cnn)
removed_file.close()


#
# plt.scatter(X_axis, Y_axis, W_size, c=x_means.labels_, s=30)
# plt.scatter(x_means.cluster_centers_[:, 0], x_means.cluster_centers_[:, 1], c="r", marker="+", s=100)
# plt.xlim(0, 3)
# plt.ylim(0, 3)
# plt.title("")

# plt.show()



# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')


# ax.scatter(X_axis, Y_axis, W_size, c='r', marker='.')
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')

# plt.show()
