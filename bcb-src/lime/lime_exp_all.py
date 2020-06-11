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
from utils import preprocess_images, transform_image, pretty_cm, evaluation_indices

# lime
import lime
from lime import lime_image

    
# creating training and test sets 
# put crater_date folder on root directory too. 
# create the training_set and test_set directories inside crater_data and put west region on training_set and other regions on test_set on each running 
# crater_data/training_set/crater/
# crater_data/training_set/non-crater/

# west region as training_set
#preprocess_images('tile2_24', 'training_set', img_dimensions=(64, 64))
#preprocess_images('tile2_25', 'training_set', img_dimensions=(64, 64))
# east region as test_set
preprocess_images('tile1_24', 'test_set', img_dimensions=(64, 64))
preprocess_images('tile1_25', 'test_set', img_dimensions=(64, 64))
preprocess_images('tile2_24', 'test_set', img_dimensions=(64, 64))
preprocess_images('tile2_25', 'test_set', img_dimensions=(64, 64))
preprocess_images('tile3_24', 'test_set', img_dimensions=(64, 64))
preprocess_images('tile3_25', 'test_set', img_dimensions=(64, 64))

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = False)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('./crater_data/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('./crater_data/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

# inspecting class labels for future reference 
labels_index = { 0 : "non-crater", 1 : "crater" }
training_set.class_indices

# initialize CNN 
# Initialising 
cnn_classifier = Sequential()

# 1st conv. layer
cnn_classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
cnn_classifier.add(MaxPooling2D(pool_size = (2, 2)))

# 2nd conv. layer
cnn_classifier.add(Conv2D(32, (3, 3), activation = 'relu')) #no need to specify the input shape
cnn_classifier.add(MaxPooling2D(pool_size = (2, 2)))

# 3nd conv. layer
cnn_classifier.add(Conv2D(64, (3, 3), activation = 'relu')) #no need to specify the input shape
cnn_classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Flattening
cnn_classifier.add(Flatten())

# Full connection
cnn_classifier.add(Dense(units = 64, activation = 'relu'))
cnn_classifier.add(Dropout(0.5)) # quite aggresive dropout, maybe reduce
cnn_classifier.add(Dense(units = 1, activation = 'sigmoid'))

cnn_classifier.summary()

# Compiling the CNN
cnn_classifier.compile(optimizer = 'adam', # 'adam'/rmsprop'
                   loss = 'binary_crossentropy', 
                   metrics = ['accuracy'])


# load the model 
cnn_classifier = keras.models.load_model('cnn_models/west_train_model_dropout.h5')

#preparing data for predictions

size = (64, 64)
X_eval = list()
y_eval = list()

# crater part
files = os.listdir('./crater_data/test_set/crater')
files.sort()

for i in range(0, len(files) - 1):
    X_eval.append(transform_image('./crater_data/test_set/crater/' + files[i + 1], size))
    y_eval.append(1)

# non-crater part
files = os.listdir('./crater_data/test_set/non-crater')
files.sort()

for i in range(0, len(files) - 1):
    X_eval.append(transform_image('./crater_data/test_set/non-crater/' + files[i + 1], size))
    y_eval.append(0)

# stacking the arrays   
X_eval = np.vstack(X_eval)

cnn_pred = cnn_classifier.predict_classes(X_eval, batch_size = 32)

print("evaluate the results using lime")
pretty_cm(cnn_pred, y_eval, ['crater', 'non-crater'])

correctly_classified_indices, misclassified_indices = evaluation_indices(cnn_pred, y_eval)

# correctly classified images
plt.figure(figsize=(50,10))
#shuffle(correctly_classified_indices)
for plot_index, good_index in enumerate(correctly_classified_indices[0:5]):
    plt.subplot(1, 5, plot_index + 1)
    plt.imshow(X_eval[good_index])
    plt.title(' Predicted: {}, Actual: {} '.format(labels_index[cnn_pred[good_index][0]], 
                                                 labels_index[y_eval[good_index]]), fontsize = 18) 
plt.savefig('results/correctly_classified.png', dpi=400)
                                                 
# lime explanation of correctly classified images
plt.figure(figsize=(50,10))
#shuffle(correctly_classified_indices)
for plot_index, good_index in enumerate(correctly_classified_indices[0:5]):
    plt.subplot(1, 5, plot_index + 1)
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(X_eval[good_index], cnn_classifier.predict_classes, top_labels=2, hide_color=0, num_samples=1000)
    temp, mask = explanation.get_image_and_mask(0, positive_only=False, num_features=1000, hide_rest=False)
    #temp, mask = explanation.get_image_and_mask(0, positive_only=True, num_features=1000, hide_rest=True)
    x = mark_boundaries(temp / 2 + 0.5, mask)
    plt.imshow(x, interpolation='none')
    plt.title(' Predicted: {}, Actual: {} '.format(labels_index[cnn_pred[good_index][0]], 
                                                 labels_index[y_eval[good_index]]), fontsize = 18)
plt.savefig('results/lime_correctly_classified.png', dpi=400)
                                                 
# misclassified images
plt.figure(figsize=(50,10))
#shuffle(misclassified_indices)
for plot_index, bad_index in enumerate(misclassified_indices[0:5]):
    plt.subplot(1, 5, plot_index + 1)
    plt.imshow(X_eval[bad_index])
    plt.title(' Predicted: {}, Actual: {} '.format(labels_index[cnn_pred[bad_index][0]], 
                                                 labels_index[y_eval[bad_index]]), fontsize = 18)
plt.savefig('results/mis_classified.png', dpi=400)
                                                 
#  lime explanation of correctly misclassified images
plt.figure(figsize=(50,10))
#shuffle(misclassified_indices)
for plot_index, bad_index in enumerate(misclassified_indices[0:5]):
    plt.subplot(1, 5, plot_index + 1)
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(X_eval[bad_index], cnn_classifier.predict_classes, top_labels=2, hide_color=0, num_samples=1000)
    temp, mask = explanation.get_image_and_mask(0, positive_only=False, num_features=1000, hide_rest=False)
    #temp, mask = explanation.get_image_and_mask(0, positive_only=True, num_features=1000, hide_rest=True)
    x = mark_boundaries(temp / 2 + 0.5, mask)
    plt.imshow(x, interpolation='none')
    plt.title(' Predicted: {}, Actual: {} '.format(labels_index[cnn_pred[bad_index][0]], 
                                                 labels_index[y_eval[bad_index]]), fontsize = 18)
plt.savefig('results/lime_mis_classified.png', dpi=400)

                                               










