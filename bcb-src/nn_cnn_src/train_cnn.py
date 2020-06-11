import os
from crater_cnn import Network
from crater_plots import plot_image, plot_conv_weights, plot_conv_layer
from crater_preprocessing import preprocess, positive_rotation_preprocess
#from blob import Blob
cwd = os.getcwd()

#preprocess('tile1_24' ,'mask' , img_dimensions=(50, 50))
#preprocess('tile1_25' ,'mask', img_dimensions=(50, 50))
#preprocess('tile2_24' ,'mask', img_dimensions=(50, 50))
#preprocess('tile2_25' ,'mask', img_dimensions=(50, 50))

from crater_loader import load_crater_data
from crater_data import Data

# Load data
images, labels, hot_one = load_crater_data('mask')
data = Data(images, hot_one, random_state=42)

print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(data.test.labels)))
print("- Validation-set:\t{}".format(len(data.validation.labels)))

model = Network(img_shape=(50, 50, 1))
model.add_convolutional_layer(5, 16)
model.add_convolutional_layer(5, 36)
model.add_flat_layer()
model.add_fc_layer(size=64, use_relu=True)
model.add_fc_layer(size=16, use_relu=True)
model.add_fc_layer(size=2, use_relu=False)
model.finish_setup()
model.set_data(data)

model_path = os.path.join(cwd, 'models', 'cnn', 'crater_model_cnn_mask.ckpt') # the models with _th indicate that they use positive samples extracted form theresholding images. 
#model.restore(model_path)

model.print_test_accuracy()

model.optimize(epochs=100)

model.save(model_path)

model.print_test_accuracy()

model.print_test_accuracy(show_example_errors=True)

model.print_test_accuracy(show_example_errors=True,
                          show_confusion_matrix=True)

image1 = data.test.images[7]
plot_image(image1)

image2 = data.test.images[14]
plot_image(image2)

weights = model.filters_weights
plot_conv_weights(weights=weights[0])
plot_conv_weights(weights=weights[1])

values = model.get_filters_activations(image1)
plot_conv_layer(values=values[0])
plot_conv_layer(values=values[1])

values = model.get_filters_activations(image2)
plot_conv_layer(values=values[0])
plot_conv_layer(values=values[1])
