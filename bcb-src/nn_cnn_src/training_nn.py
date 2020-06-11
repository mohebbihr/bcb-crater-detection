import time
from datetime import timedelta
from crater_data import Data
from crater_loader import load_crater_data_wrapper, load_crater_data
from crater_nn import Network
import numpy as np
import matplotlib.pyplot as plt
import pickle

images, labels, hot_one = load_crater_data()
data = Data(images, hot_one, random_state=42, build_images_col=True)
tr_d = list(zip(data.train.images_col, data.train.cls))
te_d = list(zip(data.test.images_col, data.test.cls))
va_d = list(zip(data.validation.images_col, data.validation.cls))

iteration = 0
experiment_data = []

input_size = 50*50

for i in range(1):
    iteration += 1

    start = time.time()
        
    # define the network shape to be used and the activation threshold
    model = Network([input_size, 8, 1], False)
    model.threshold = 0.3

    # the schedule is how the learning rate will be
    # changed during the training
    epochs = 100
    schedule = [(0.1)*(0.5)**np.floor(float(i)/(30)) for i in range(epochs)]
    #schedule = np.linspace(0.5, 0.01, epochs)
    for eta in schedule:
        # the total epochs is given by the schedule loop
        # we chose minibatch size to be 3
        model.SGD(tr_d, 1, 3, eta, te_d)

    end = time.time()

    # After training is complete, store this model training history
    # to the experiment data
    experiment_data.append(np.array(model.history))
    
    # store current results data to disk
    np.save("experiment_data", experiment_data)
    
    # save current model to disk
    with open('results/models/crater_nn_model_%05d_%02d.pkl' % (input_size, i), 'wb') as output:
        pickle.dump(model, output)

    elapsed_time = end - start
    print (iteration, timedelta(seconds=elapsed_time))
