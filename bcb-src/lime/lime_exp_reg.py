# Loading libraries ----

# misc
import os
import glob
import shutil
from random import sample, randint, shuffle
import numpy as np
import pandas as pd
from skimage.segmentation import mark_boundaries

# sci-kit learn
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# plots
import matplotlib.pyplot as plt
import seaborn as sns

# image operation
import cv2
from PIL import Image

# keras 
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.applications import inception_v3 as inc_net

# lime
import lime
from lime import lime_image

#def create_necc_dirc()
    # this function remove old training and test sets
    # create new directories if necessary
    
    
def preprocess_images(filename, dest_folder, img_dimensions=(64, 64)):
    # function to pre-process images and create training and test sets. 
    src = os.path.join('crater_data', 'images', filename)
    dst = os.path.join('crater_data', dest_folder)
    tgt_height, tgt_width = img_dimensions

    # create new directories if necessary
    for imgtype in ['crater', 'non-crater']:
        tgdir = os.path.join(dst, imgtype)
        if not os.path.isdir(tgdir):
            os.makedirs(tgdir)

    for src_filename in glob.glob(os.path.join(src, '*', '*.jpg')):
        #print(src_filename)
        pathinfo = src_filename.split(os.path.sep)
        img_type = pathinfo[-2] # crater or non-crater
        filename = pathinfo[-1] # the actual name of the jpg

        dst_filename = os.path.join(dst, img_type, filename)

        # read the original image and get size info
        src_img = cv2.imread(src_filename)

        # resize image, normalize and write to disk
        scaled_img = cv2.resize(src_img, (tgt_height, tgt_width))
        cv2.normalize(scaled_img, scaled_img, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(dst_filename, scaled_img)
    
    print(dest_folder + " Done!")
    
def move_random_files(path_from, path_to, n):
    # function for moving random files from one directory to another (used for creating train and test set)
    files = os.listdir(path_from)
    files.sort()
    files = files[1:] #omiting .DS_Store

    for i in sample(range(0, len(files)-1), n):
        f = files[i]
        src = path_from + f
        dst = path_to + f
        shutil.move(src, dst)
        
def preview_random_image(path):
    # function for previewing a random image from a given directory
    files = os.listdir(path)
    files.sort()
    img_name = files[randint(1, len(files) - 1)]
    img_preview_name = path + img_name
    image = Image.open(img_preview_name)
    plt.imshow(image)
    plt.title(img_name)
    plt.show()
    width, height = image.size
    print ("Dimensions:", image.size, "Total pixels:", width * height)
    
def pretty_cm(y_pred, y_truth, labels):
    # pretty implementation of a confusion matrix
    cm = metrics.confusion_matrix(y_truth, y_pred)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, fmt="d", linewidths=.5, square = True, cmap = 'BuGn_r')
    # labels, title and ticks
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('Actual label')
    ax.set_title('Accuracy: {0}'.format(metrics.accuracy_score(y_truth, y_pred)), size = 15) 
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
    plt.savefig('results/regression/confusion_matrix.png', dpi=400)
    
    
def img_to_1d_greyscale(img_path, size):
    # function for loading, resizing and converting an image into greyscale
    # used for logistic regression
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, size)
    return(pd.Series(img.flatten()))

def show_image(image):
    # function for viewing an image
    fig = plt.figure(figsize = (5, 25))
    ax = fig.add_subplot(111)
    ax.imshow(image, interpolation='none')
    plt.show()

def transform_image(path, size):
    # function for transforming images into a format supported by CNN
    x = load_img(path, target_size=(size[0], size[1]))
    x = img_to_array(x) / 255
    x = np.expand_dims(x, axis=0)
    return (x)
    
def evaluation_indices(y_pred, y_test):
    # function for getting correctly and incorrectly classified indices
    index = 0
    correctly_classified_indices = []
    misclassified_indices = []
    for label, predict in zip(y_test, y_pred):
        if label != predict: 
            misclassified_indices.append(index)
        else:
            correctly_classified_indices.append(index)
        index +=1
    return (correctly_classified_indices, misclassified_indices)
    
# creating training and test sets 
# put crater_date folder on root directory too. 
# create the training_set and test_set directories inside crater_data and put west region on training_set and other regions on test_set on each running 
# crater_data/training_set/crater/
# crater_data/training_set/non-crater/

# west region as training_set
preprocess_images('tile2_24', 'training_set', img_dimensions=(64, 64))
preprocess_images('tile2_25', 'training_set', img_dimensions=(64, 64))
# east region as test_set
preprocess_images('tile1_24', 'test_set', img_dimensions=(64, 64))
preprocess_images('tile1_25', 'test_set', img_dimensions=(64, 64))
#preprocess_images('tile2_24', 'test_set', img_dimensions=(64, 64))
#preprocess_images('tile2_25', 'test_set', img_dimensions=(64, 64))
#preprocess_images('tile3_24', 'test_set', img_dimensions=(64, 64))
#preprocess_images('tile3_25', 'test_set', img_dimensions=(64, 64))

size = (64, 64)
labels_index = { 0 : "non-crater", 1 : "crater" }

# defining empty containers
X_train = pd.DataFrame(np.zeros((8000, size[0] * size[1])))
X_test = pd.DataFrame(np.zeros((2000, size[0] * size[1])))
y_train = list()
y_test = list()

counter_train = 0
counter_test = 0

# training set ----

files = os.listdir('./crater_data/training_set/crater')
files.sort()

for i in range(1, len(files)):
    X_train.iloc[counter_train, :] = img_to_1d_greyscale('./crater_data/training_set/crater/' + files[i], size) / 255
    y_train.append(1)
    counter_train += 1
    
files = os.listdir('./crater_data/training_set/non-crater')
files.sort()

for i in range(1, len(files)):
    X_train.iloc[counter_train, :] = img_to_1d_greyscale('crater_data/training_set/non-crater/' + files[i], size) / 255
    y_train.append(0)
    counter_train += 1
   
# training set ----

files = os.listdir('./crater_data/test_set/crater')
files.sort()

for i in range(1, len(files)):
    X_test.iloc[counter_test, :] = img_to_1d_greyscale('crater_data/test_set/crater/' + files[i], size) / 255
    y_test.append(1)
    counter_test += 1
    
files = os.listdir('./crater_data/test_set/non-crater')
files.sort()

for i in range(1, len(files)):
    X_test.iloc[counter_test, :] = img_to_1d_greyscale('crater_data/test_set/non-crater/' + files[i], size) / 255
    y_test.append(0)  
    counter_test += 1
    
#preparing data for predictions

size = (64, 64)
X_eval = list()
y_eval = list()

# crater part
files = os.listdir('./crater_data/test_set/crater')
files.sort()

for i in range(0, len(files) - 1):
    X_eval.append(transform_image('./crater_data/test_set/crater/' + files[i + 1], size))
    y_eval.append(0)

# non-crater part
files = os.listdir('./crater_data/test_set/non-crater')
files.sort()

for i in range(0, len(files) - 1):
    X_eval.append(transform_image('./crater_data/test_set/non-crater/' + files[i + 1], size))
    y_eval.append(1)

# stacking the arrays   
X_eval = np.vstack(X_eval)
    
logreg_classifier = LogisticRegression(solver = 'lbfgs')

logreg_classifier.fit(X_train, y_train)

logreg_pred = logreg_classifier.predict(X_test)

pretty_cm(logreg_pred, y_test, ['crater', 'non-crater'])

correctly_classified_indices, misclassified_indices = evaluation_indices(logreg_pred, y_test)

for plot_index, bad_index in enumerate(misclassified_indices[0:5]):
    plt.subplot(1, 5, plot_index + 1)
    plt.imshow(np.reshape(X_test.iloc[bad_index, :].values, size))
    plt.title('Predicted: {}, Actual: {}'.format(labels_index[logreg_pred[bad_index]], 
                                                 labels_index[y_test[bad_index]]), fontsize = 15)
plt.savefig('results/regression/correctly_classified.png', dpi=400)
                                                 
# lime explanation of correctly classified images
plt.figure(figsize=(25,5))
shuffle(correctly_classified_indices)
for plot_index, good_index in enumerate(correctly_classified_indices[0:5]):
    plt.subplot(1, 5, plot_index + 1)
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(X_eval[good_index], logreg_classifier.predict, top_labels=2, hide_color=0, num_samples=1000)
    temp, mask = explanation.get_image_and_mask(0, positive_only=False, num_features=10, hide_rest=False)
    x = mark_boundaries(temp / 2 + 0.5, mask)
    plt.imshow(x, interpolation='none')
    plt.title('Predicted: {}, Actual: {}'.format(labels_index[logreg_pred[good_index][0]], 
                                                 labels_index[y_eval[good_index]]), fontsize = 15)
plt.savefig('results/regression/lime_correctly_classified.png', dpi=400)
                                                 
# misclassified images
plt.figure(figsize=(25,5))
shuffle(misclassified_indices)
for plot_index, bad_index in enumerate(misclassified_indices[0:5]):
    plt.subplot(1, 5, plot_index + 1)
    plt.imshow(X_eval[bad_index])
    plt.title('Predicted: {}, Actual: {}'.format(labels_index[logreg_pred[bad_index][0]], 
                                                 labels_index[y_eval[bad_index]]), fontsize = 15)
plt.savefig('results/regression/mis_classified.png', dpi=400)
                                                 
#  lime explanation of correctly misclassified images
plt.figure(figsize=(25,5))
shuffle(misclassified_indices)
for plot_index, bad_index in enumerate(misclassified_indices[0:5]):
    plt.subplot(1, 5, plot_index + 1)
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(X_eval[bad_index], logreg_classifier.predict, top_labels=2, hide_color=0, num_samples=1000)
    temp, mask = explanation.get_image_and_mask(0, positive_only=False, num_features=10, hide_rest=False)
    x = mark_boundaries(temp / 2 + 0.5, mask)
    plt.imshow(x, interpolation='none')
    plt.title('Predicted: {}, Actual: {}'.format(labels_index[logreg_pred[bad_index][0]], 
                                                 labels_index[y_eval[bad_index]]), fontsize = 15)
plt.savefig('results/regression/lime_mis_classified.png', dpi=400)

