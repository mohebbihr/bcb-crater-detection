import os
from crater_cnn import Network
from crater_plots import plot_image, plot_conv_weights, plot_conv_layer
cwd = os.getcwd()

#preprocess(img_dimensions=(30, 30))

from crater_loader import load_crater_data
from crater_data import Data

# Load data
images, labels, hot_one = load_crater_data()
data = Data(images, hot_one, random_state=42)

model = Network(img_shape=(30, 30, 1))
model.add_convolutional_layer(5, 16)
model.add_convolutional_layer(5, 36)
model.add_flat_layer()
model.add_fc_layer(size=128, use_relu=True)
model.add_fc_layer(size=2, use_relu=False)
model.finish_setup()
model.set_data(data)

model_path = os.path.join(cwd, 'model.ckpt')
model.restore(model_path)

image1 = data.test.images[7]
image2 = data.test.images[14]

print(model.predict([image1]))
print(model.predict([image1, image2]))

samples = [image1, image2]
print(model.predict(samples))

result = data.test.cls[0], data.test.labels[0], model.predict([data.test.images[0]])
print(result)
