from skimage.transform import pyramid_gaussian
import cv2 as cv
from helper import sliding_window
import time
import os
import csv
from crater_cnn import Network as CNN
from crater_nn import Network as NN
import pickle
cwd = os.getcwd()

# setup CNN
cnn = CNN(img_shape=(50, 50, 1))
cnn.add_convolutional_layer(5, 16)
cnn.add_convolutional_layer(5, 36)
cnn.add_flat_layer()
cnn.add_fc_layer(size=64, use_relu=True)
cnn.add_fc_layer(size=16, use_relu=True)
cnn.add_fc_layer(size=2, use_relu=False)
cnn.finish_setup()
# model.set_data(data)

# setup NN
nn = CNN(img_shape=(50, 50, 1))
nn.add_flat_layer()
nn.add_fc_layer(size=50 * 50, use_relu=True)
nn.add_fc_layer(size=16, use_relu=True)
nn.add_fc_layer(size=2, use_relu=False)
nn.finish_setup()

# restore previously trained CNN model
cnn_model_path = os.path.join(cwd, 'results/models/crater_west_model_cnn.ckpt')
cnn.restore(cnn_model_path)

# restore previously trained NN model
nn_model_path = os.path.join(cwd, 'results/models/crater_west_model_nn.ckpt')
nn.restore(nn_model_path)

#with open('results/models/nn_model_02500_00.pkl', 'rb') as finput:
#    nn = pickle.load(finput)

path = os.path.join('crater_data', 'images')
img = cv.imread(os.path.join(path, 'tile2_24.pgm'), 0)
img = cv.normalize(img, img, 0, 255, cv.NORM_MINMAX)/255.0

crater_list_cnn = []
crater_list_nn = []

win_sizes = range(20, 30, 2)
# loop over the image pyramid
for (i, resized) in enumerate(pyramid_gaussian(img, downscale=1.5)):
    if resized.shape[0] < 31:
        break
    for winS in win_sizes:
        print("Resized shape: %d, Window size: %d" % (resized.shape[0], winS))

        # loop over the sliding window for each layer of the pyramid
        # this process takes about 7 hours. To do quick test, we may try stepSize
        # to be large (60) and see if code runs OK
        #for (x, y, window) in sliding_window(resized, stepSize=2, windowSize=(winS, winS)):
        for (x, y, window) in sliding_window(resized, stepSize=60, windowSize=(winS, winS)):
            # since we do not have a classifier, we'll just draw the window
            clone = resized.copy()
            y_b = y + winS
            x_r = x + winS
            crop_img = clone[y:y_b, x:x_r]
            crop_img =cv.resize(crop_img, (50, 50))
            crop_img = crop_img.flatten()
            
            p_non, p_crater = cnn.predict([crop_img])[0]
            p_nn_non, nn_p = nn.predict([crop_img])[0]
            #nn_p = nn.feedforward_flat(crop_img)[0,0]
            
            scale_factor = 1.5 ** i
            if p_crater >= 0.5 or nn_p >= 0.5:
                x_c = int((x + 0.5 * winS) * scale_factor)
                y_c = int((y + 0.5 * winS) * scale_factor)
                crater_size = int(winS * scale_factor)
                
                if p_crater >= 0.5:
                    crater_data = [x_c, y_c, crater_size, p_crater, 1]
                    crater_list_cnn.append(crater_data)
                if nn_p >= 0.5:
                    crater_data = [x_c, y_c, crater_size, nn_p, 1]
                    crater_list_nn.append(crater_data)
            
            # if we want to see where is processed.
            # cv.rectangle(clone, (x, y), (x + winS, y + winS), (0, 255, 0), 2)
            # cv.imshow("Window", clone)
            # cv.waitKey(1)
cnn_file = open("results/west_train_center_test_2_24_cnn.csv","w")
with cnn_file:
    writer = csv.writer(cnn_file, delimiter=',')
    writer.writerows(crater_list_cnn)
cnn_file.close()

nn_file = open("results/west_train_crater_test_2_24_nn.csv","w")
with nn_file:
    writer = csv.writer(nn_file, delimiter=',')
    writer.writerows(crater_list_nn)
nn_file.close()


print("CNN detected ", len(crater_list_cnn), "craters")

print("NN detected ", len(crater_list_nn), "craters")

