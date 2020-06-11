import os
from crater_cnn import Network
from crater_plots import plot_image, plot_conv_weights, plot_conv_layer
from crater_preprocessing import preprocess
from sklearn.model_selection import KFold
cwd = os.getcwd()

# This file represents the 10 fold cross validation experiment for neural network with FC layers over the entire dataset.
# 1- remove all the data from normalized images folder.
# 2- pre-process images for each region
# 3- perform 10 fold cv for a region and save the model.

# preprocess the west region images (tile1_24, tile1_25)
preprocess('tile1_24', img_dimensions=(50, 50))
preprocess('tile1_25', img_dimensions=(50, 50))
preprocess('tile2_24', img_dimensions=(50, 50))
preprocess('tile2_25', img_dimensions=(50, 50))
preprocess('tile3_24', img_dimensions=(50, 50))
preprocess('tile3_25', img_dimensions=(50, 50))

from crater_loader import load_crater_data
from crater_data import KCV_Data

# Load data
images, labels, hot_one = load_crater_data()

# define model
model = Network(img_shape=(50, 50, 1))
model.add_convolutional_layer(5, 16)
model.add_convolutional_layer(5, 36)
model.add_flat_layer()
model.add_fc_layer(size=64, use_relu=True)
model.add_fc_layer(size=16, use_relu=True)
model.add_fc_layer(size=2, use_relu=False)
model.finish_setup()

# perform k fold cross validation
kf = KFold(n_splits=5)
i = 1
f1_avg = 0.0
acc_avg = 0.0

for train_index, test_index in kf.split(images):
    X_train, X_test = images[train_index], images[test_index]
    Y_train, Y_test = hot_one[train_index], hot_one[test_index]

    data = KCV_Data(X_train, X_test, Y_train, Y_test)
    
    print("fold: " + str(i))

    model.set_data(data)
    model.optimize_no_valid(epochs=20)
    
    # get f1 and acc measures.
    acc , f1 = model.print_test_accuracy()
    
    acc_avg += acc
    f1_avg += f1
    
    print(" Acc : {0: .1%} , F1 : {1: .1%}".format(acc, f1))
    
    i += 1

f1_avg /= 5.0
acc_avg /= 5.0
print("5-fold Acc : {0: .1%} , F1 : {1: .1%}".format(acc_avg, f1_avg))

model_path = os.path.join(cwd, 'results', 'models', 'crater_all_5fcv_cnn.ckpt')
#model.restore(model_path)

model.save(model_path)

model.print_test_accuracy(show_example_errors=True)

model.print_test_accuracy(show_example_errors=True,
                          show_confusion_matrix=True)


