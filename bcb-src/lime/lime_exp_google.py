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

from keras.applications.inception_v3 import InceptionV3
from keras.applications import ResNet50
from keras.applications import VGG16
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.optimizers import SGD

import argparse
from skimage.segmentation import mark_boundaries
from keras.applications.imagenet_utils import decode_predictions
   
ap = argparse.ArgumentParser()
ap.add_argument("-model", "--model", type=str, default="vgg16", help="name of pre-trained network to use")
ap.add_argument("-path", "--path", type=str , help="path to the model to load")
args = vars(ap.parse_args())

MODELS = {
	"vgg16": VGG16,
	"inception": InceptionV3,
	"resnet": ResNet50
}
 
# creating training and test sets 
# put crater_date folder on root directory too. 
# create the training_set and test_set directories inside crater_data and put west region on training_set and other regions on test_set on each running 
# crater_data/training_set/crater/
# crater_data/training_set/non-crater/

# west region as training_set
#preprocess_images('tile2_24', 'training_set', img_dimensions=(64, 64))
#preprocess_images('tile2_25', 'training_set', img_dimensions=(64, 64))
# east region as test_set
#preprocess_images('tile1_24', 'test_set', img_dimensions=(64, 64))
#preprocess_images('tile1_25', 'test_set', img_dimensions=(64, 64))
#preprocess_images('tile2_24', 'test_set', img_dimensions=(64, 64))
#preprocess_images('tile2_25', 'test_set', img_dimensions=(64, 64))
#preprocess_images('tile3_24', 'test_set', img_dimensions=(64, 64))
#preprocess_images('tile3_25', 'test_set', img_dimensions=(64, 64))

input_shape = (224, 224)
preprocess = imagenet_utils.preprocess_input

if args["model"] in ("inception"):
	input_shape = (299, 299)
	preprocess = preprocess_input

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

print("[INFO] loading {}...".format(args["path"]))
Network = MODELS[args["model"]]
base_model = Network(weights="imagenet", include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# let's add a fully-connected layer
x = Dense(512, activation='relu')(x)

# and a logistic layer -- we have 2 classes
predictions = Dense(units = 1, activation = 'sigmoid')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='binary_crossentropy',metrics = ['accuracy'])
model.fit_generator(training_set, steps_per_epoch=400, epochs=3)

for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

for layer in model.layers:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='binary_crossentropy',metrics = ['accuracy'])

# we train our model again (this time fine-tuning all layers
model.fit_generator(training_set, steps_per_epoch=400, epochs=20)

model.save('deep_models/inception_west_train.h5')

model = load_model('deep_models/inception_west_train.h5')

#preparing data for predictions
labels_index = { 0 : "non-crater", 1 : "crater" }
X_eval = list()
y_eval = list()

# crater part
files = os.listdir('./crater_data/test_set/crater')
files.sort()

for i in range(0, len(files) - 1):
    X_eval.append(transform_image('./crater_data/test_set/crater/' + files[i + 1], input_shape))
    y_eval.append(1)

# non-crater part
files = os.listdir('./crater_data/test_set/non-crater')
files.sort()

for i in range(0, len(files) - 1):
    X_eval.append(transform_image('./crater_data/test_set/non-crater/' + files[i + 1], input_shape))
    y_eval.append(0)

# stacking the arrays   
X_eval = np.vstack(X_eval)

preds = model.predict(X_eval, verbose=1)

print("evaluate the results using lime")
pretty_cm(preds, y_eval, ['crater', 'non-crater'], 'results/googlenet/confusion_matrix.png')

correctly_classified_indices, misclassified_indices = evaluation_indices(preds, y_eval)

# correctly classified images
plt.figure(figsize=(50,10))
#shuffle(correctly_classified_indices)
for plot_index, good_index in enumerate(correctly_classified_indices[0:5]):
    plt.subplot(1, 5, plot_index + 1)
    plt.imshow(X_eval[good_index])
    plt.title(' Predicted: {}, Actual: {} '.format(labels_index[preds[good_index][0]], 
                                                 labels_index[y_eval[good_index]]), fontsize = 18) 
plt.savefig('results/googlenet/correctly_classified.png', dpi=400)
                                                 
# lime explanation of correctly classified images
plt.figure(figsize=(50,10))
#shuffle(correctly_classified_indices)
for plot_index, good_index in enumerate(correctly_classified_indices[0:5]):
    plt.subplot(1, 5, plot_index + 1)
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(X_eval[good_index], model.predict, top_labels=2, hide_color=0, num_samples=1000)
    temp, mask = explanation.get_image_and_mask(0, positive_only=False, num_features=1000, hide_rest=False)
    #temp, mask = explanation.get_image_and_mask(0, positive_only=True, num_features=1000, hide_rest=True)
    x = mark_boundaries(temp / 2 + 0.5, mask)
    plt.imshow(x, interpolation='none')
    plt.title(' Predicted: {}, Actual: {} '.format(labels_index[preds[good_index][0]], 
                                                 labels_index[y_eval[good_index]]), fontsize = 18)
plt.savefig('results/googlenet/lime_correctly_classified.png', dpi=400)
                                                 
# misclassified images
plt.figure(figsize=(50,10))
#shuffle(misclassified_indices)
for plot_index, bad_index in enumerate(misclassified_indices[0:5]):
    plt.subplot(1, 5, plot_index + 1)
    plt.imshow(X_eval[bad_index])
    plt.title(' Predicted: {}, Actual: {} '.format(labels_index[preds[bad_index][0]], 
                                                 labels_index[y_eval[bad_index]]), fontsize = 18)
plt.savefig('results/googlenet/mis_classified.png', dpi=400)
                                                 
#  lime explanation of correctly misclassified images
plt.figure(figsize=(50,10))
#shuffle(misclassified_indices)
for plot_index, bad_index in enumerate(misclassified_indices[0:5]):
    plt.subplot(1, 5, plot_index + 1)
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(X_eval[bad_index], model.predict, top_labels=2, hide_color=0, num_samples=1000)
    temp, mask = explanation.get_image_and_mask(0, positive_only=False, num_features=1000, hide_rest=False)
    #temp, mask = explanation.get_image_and_mask(0, positive_only=True, num_features=1000, hide_rest=True)
    x = mark_boundaries(temp / 2 + 0.5, mask)
    plt.imshow(x, interpolation='none')
    plt.title(' Predicted: {}, Actual: {} '.format(labels_index[preds[bad_index][0]], 
                                                 labels_index[y_eval[bad_index]]), fontsize = 18)
plt.savefig('results/googlenet/lime_mis_classified.png', dpi=400)

                                               










