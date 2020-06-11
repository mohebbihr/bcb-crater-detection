# organize imports
import os
import glob
import datetime
import json
from crater_preprocessing import preprocess

# load the user configs
with open('conf/conf.json') as f:    
	config = json.load(f)

# config variables
model_name = config["model"]
 
if model_name == "vgg16":
	image_size = (224, 224)
elif model_name == "vgg19":
	image_size = (224, 224)
elif model_name == "resnet50":
	image_size = (224, 224)
elif model_name == "inceptionv3":
	image_size = (299, 299)
elif model_name == "inceptionresnetv2":
	image_size = (299, 299)
elif model_name == "mobilenet":
	image_size = (224, 224)
elif model_name == "xception":
	image_size = (299, 299)
else:
	image_size = (50, 50)

# use west region as training set. 
preprocess('tile1_24' , 'train',img_dimensions=image_size)
preprocess('tile1_25' , 'train', img_dimensions=image_size)

# use center region as test
preprocess('tile1_24' , 'test',img_dimensions=image_size)
preprocess('tile1_25' , 'test', img_dimensions=image_size)

# print end time
print ("pre-processing end for model: " + model_name)