import os
from crater_cnn import Network
from crater_plots import plot_image, plot_conv_weights, plot_conv_layer
from crater_preprocessing import preprocess
cwd = os.getcwd()

# preprocess the west region images (tile1_24, tile1_25)
#preprocess('tile3_24' , img_dimensions=(50, 50))
#preprocess('tile3_25' , img_dimensions=(50, 50))

from crater_loader import load_crater_data
from crater_data import Data

# Load dataxcv
images, labels, hot_one = load_crater_data()
data = Data(images, hot_one, random_state=42)

print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(data.test.labels)))
print("- Validation-set:\t{}".format(len(data.validation.labels)))

model = Network(img_shape=(50, 50, 1))
model.add_flat_layer()
model.add_fc_layer(size=50 * 50, use_relu=True)
model.add_fc_layer(size=16, use_relu=True)
model.add_fc_layer(size=2, use_relu=False)
model.finish_setup()
#model.set_data(data)

cnn = Network(img_shape=(50, 50, 1))
cnn.add_convolutional_layer(5, 16)
cnn.add_convolutional_layer(5, 36)
cnn.add_flat_layer()
cnn.add_fc_layer(size=64, use_relu=True)
cnn.add_fc_layer(size=16, use_relu=True)
cnn.add_fc_layer(size=2, use_relu=False)
cnn.finish_setup()

model_path = os.path.join(cwd, 'results', 'nn_models', 'crater_east_model_nn.ckpt')
model.restore(model_path)

cnn_model_path = os.path.join(cwd, 'results/models/crater_west_model_cnn.ckpt')
cnn.restore(cnn_model_path)

image1 = data.test.images[7]
p_nn_non, nn_p = model.predict([image1])[0]

p_non, p_crater = cnn.predict([image1])[0]

print("nn predict: " + str(nn_p))

print("cnn predict: " + str(cnn_p))

#model.print_test_accuracy()

#model.optimize(epochs=20)

#model.save(model_path)

#model.print_test_accuracy()

#model.print_test_accuracy(show_example_errors=True)

#model.print_test_accuracy(show_example_errors=True, show_confusion_matrix=True)

#image1 = data.test.images[7]
#plot_image(image1)

#image2 = data.test.images[14]
#plot_image(image2)

#weights = model.filters_weights
#plot_conv_weights(weights=weights[0])
#plot_conv_weights(weights=weights[1])

#values = model.get_filters_activations(image1)
#plot_conv_layer(values=values[0])
#plot_conv_layer(values=values[1])

#values = model.get_filters_activations(image2)
#plot_conv_layer(values=values[0])
#plot_conv_layer(values=values[1])
