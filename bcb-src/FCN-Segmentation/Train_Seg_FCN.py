import cv2, os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import keras, time, warnings
from keras.models import *
from keras.layers import *
from sklearn.utils import shuffle
import scipy
from keras import optimizers
import imutils
import glob
from keras.callbacks import ModelCheckpoint

cwd = os.getcwd()


# task: think about adding rotattions of images into training set.
# input: tileimg should be something like tile1_24. 
def make_train_set(tileimg):
    
    src_org_list = [os.path.join('crater_data', 'slices','org_noverlap', tileimg), os.path.join('crater_data', 'slices','org_24overlap', tileimg)]
    src_mask_list = [os.path.join('crater_data', 'slices','mask_noverlap', tileimg), os.path.join('crater_data', 'slices','mask_24overlap', tileimg)]
    
    dst = os.path.join('crater_data', 'slices', 'train')
    counter = 0
    
    
    # create new directories if necessary
    for imgtype in ['org', 'mask']:
        tgdir = os.path.join(dst, imgtype)
        if not os.path.isdir(tgdir):
            os.makedirs(tgdir)
            
    for i in range(0,2): 
        
        src_org = src_org_list[i]
        src_mask = src_mask_list[i]
        # add original samples to train folder.
        for src_filename in glob.glob(os.path.join(src_org, '*.jpg')):
            
             # read the original image and get size info
            src_img = cv2.imread(src_filename)
            pathinfo = src_filename.split(os.path.sep)
            img_type = pathinfo[-2] # org or mask
            filename = pathinfo[-1] # the actual name of the jpg
            
            #print("img_type: " + img_type);
            
            rotated90 = imutils.rotate_bound(src_img, 90)
            rotated180 = imutils.rotate_bound(src_img, 180)
            rotated270 = imutils.rotate_bound(src_img, 270)
            
            dst_filename = os.path.join(dst, 'org', '0_'+filename)
            dst_filename90 = os.path.join(dst, 'org', '90_'+filename)
            dst_filename180 = os.path.join(dst, 'org', '180_'+filename)
            dst_filename270 = os.path.join(dst, 'org', '270_'+filename)
            
            # normalizing the image?? No. The model requires a 3 channel picture.
            cv2.imwrite(dst_filename, src_img)
            cv2.imwrite(dst_filename90, rotated90)
            cv2.imwrite(dst_filename180, rotated180)
            cv2.imwrite(dst_filename270, rotated270)
            
            counter += 4
            
            
        # add mask samples to train folder.
        for src_filename in glob.glob(os.path.join(src_mask, '*.jpg')):
             # read the original image and get size info
            src_img = cv2.imread(src_filename)
            pathinfo = src_filename.split(os.path.sep)
            img_type = pathinfo[-2] # org or mask
            filename = pathinfo[-1] # the actual name of the jpg
            
            rotated90 = imutils.rotate_bound(src_img, 90)
            rotated180 = imutils.rotate_bound(src_img, 180)
            rotated270 = imutils.rotate_bound(src_img, 270)
            
            dst_filename = os.path.join(dst, 'mask', '0_'+filename)
            dst_filename90 = os.path.join(dst, 'mask', '90_'+filename)
            dst_filename180 = os.path.join(dst, 'mask', '180_'+filename)
            dst_filename270 = os.path.join(dst,  'mask', '270_'+filename)
            
            # normalizing the image??
            cv2.imwrite(dst_filename, src_img)
            cv2.imwrite(dst_filename90, rotated90)
            cv2.imwrite(dst_filename180, rotated180)
            cv2.imwrite(dst_filename270, rotated270)
            
            counter += 4
        
    for imgtype in ['org', 'mask']:
        tgdir = os.path.join(dst,imgtype,'.DS_Store')
        if os.path.exists(tgdir):
            os.remove(tgdir)
        
    print("Adding "+str(counter)+" org and mask samples of "+tileimg+" to trainining set.")

def extract_data_adds(dir_data):
	# The function extracts the image and mask addresses
    folders = os.listdir(dir_data)	# directory where data was stored.

    img_folders = [f for f in folders if f[:3] == 'org'] # image folders start by "org' prefix
    seg_folders = [f for f in folders if f[:4] == 'mask']    # mask folders start by 'mask' prefix
    if(len(img_folders) == 0):
		input('Warning: Train directory is empty')    # if the directory is empty print the warning

    img_adds = []  # a list of address of images
    seg_adds = []  # a list of address of masks
    for i in range(len(img_folders)):  # for any folder read the subfolder


        img_folder = os.path.join(dir_data, img_folders[i])   # list of images subfolders
        seg_folder = dir_data + seg_folders[i]    #list of mask subfolders
        print('img_folder: ' + str(img_folder))
        img_files = os.listdir(img_folder)    # list of images in subfolder
        seg_files = os.listdir(seg_folder)    # list of masks in subfolder

        for j in range(len(img_files)):   #append images and masks to the list of files
            img_adds.append(os.path.join(img_folder,img_files[j]))
            seg_adds.append(os.path.join(seg_folder,seg_files[j]))
	return img_adds, seg_adds

##########################################################

def getImageArr( path , width , height ):
        # the function reads the image from the directory and return the resized version of the image.
        img = cv2.imread(path, 1)
        img = np.float32(cv2.resize(img, ( width , height ))) / 127.5 - 1
        return img

def getSegmentationArr( path , nClasses ,  width , height  ):
    # the function reads a mask from the directory and generate the seg_labels with the size (widthxheightsx2) which 2 is the number of classes

    seg_labels = np.zeros((  height , width  , nClasses ))
    img = cv2.imread(path, 1)
    img = cv2.resize(img, ( width , height ))
    img = img[:, : , 0]

    seg_labels[: , : , 0] = (img == 0).astype(int)
    seg_labels[: , : , 1] = (np.ones(img.shape) - seg_labels[: , : , 0]).astype(int)

    return seg_labels


def extract_data(X_add, Y_add, input_width, input_height):
    # the function receives two list of image and mask addresses and return all the images and masks in two 3-D array; X, Y
	X = [] # list of images
	Y = [] # list of masks

	for img_add, seg_add in zip(X_add, Y_add):
		X.append( getImageArr(img_add , input_width , input_height))
		Y.append( getSegmentationArr(seg_add, 2 , input_width , input_height))
	X, Y = np.array(X) , np.array(Y)
	return X, Y

##############################################################

def give_color_to_seg_img(seg,n_classes):
    # generate a color image based on the segmented image.
    
    if len(seg.shape)==3:
        seg = seg[:,:,0]
    seg_img = np.zeros( (seg.shape[0],seg.shape[1],3) ).astype('float')
    colors = sns.color_palette("hls", n_classes)
    
    for c in range(n_classes):
        segc = (seg == c)
        seg_img[:,:,0] += (segc*( colors[c][0] ))
        seg_img[:,:,1] += (segc*( colors[c][1] ))
        seg_img[:,:,2] += (segc*( colors[c][2] ))

    return(seg_img)

def FCN8_custom( nClasses ,  input_height, input_width):
    ## input_height and width must be devisible by 32 because maxpooling with filter size = (2,2) is operated 5 times,
    ## which makes the input_height and width 2^5 = 32 times smaller
    assert input_height%32 == 0
    assert input_width%32 == 0
    IMAGE_ORDERING =  "channels_last" 

    img_input = Input(shape=(input_height,input_width, 3)) ## Assume 224,224,3
    
    ## Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format=IMAGE_ORDERING )(img_input)
    tmp = Conv2D(64, (3, 3), activation='relu', padding='same', name='blocktmp_conv1', data_format=IMAGE_ORDERING )(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format=IMAGE_ORDERING )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format=IMAGE_ORDERING )(x)
    f1 = x
    
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format=IMAGE_ORDERING )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format=IMAGE_ORDERING )(x)
    f2 = x

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format=IMAGE_ORDERING )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format=IMAGE_ORDERING )(x)
    pool3 = x

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', data_format=IMAGE_ORDERING )(x)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format=IMAGE_ORDERING )(x)## (None, 14, 14, 512) 

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format=IMAGE_ORDERING )(pool4)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format=IMAGE_ORDERING )(x)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format=IMAGE_ORDERING )(x)## (None, 7, 7, 512)


    ######################################
    n = 4096
    o = ( Conv2D( n , ( 7 , 7 ) , activation='relu' , padding='same', name="conv6", data_format=IMAGE_ORDERING))(pool5)
    conv7 = ( Conv2D( n , ( 1 , 1 ) , activation='relu' , padding='same', name="conv7", data_format=IMAGE_ORDERING))(o)
    conv7_4 = Conv2DTranspose( nClasses , kernel_size=(16,16) ,  strides=(16,16) , use_bias=False, data_format=IMAGE_ORDERING )(conv7)

    pool411 = ( Conv2D( nClasses , ( 1 , 1 ) , activation='relu' , padding='same', name="pool4_11", data_format=IMAGE_ORDERING))(pool4)
    pool411_2 = (Conv2DTranspose( nClasses , kernel_size=(8,8) ,  strides=(8,8) , use_bias=False, data_format=IMAGE_ORDERING ))(pool411)

    pool311 = ( Conv2D( nClasses , ( 1 , 1 ) , activation='relu' , padding='same', name="pool3_11", data_format=IMAGE_ORDERING))(pool3)
    pool311_2 = (Conv2DTranspose( nClasses , kernel_size=(4,4) ,  strides=(4,4) , use_bias=False, data_format=IMAGE_ORDERING ))(pool311)

    pool211 = ( Conv2D( nClasses , ( 1 , 1 ) , activation='relu' , padding='same', name="pool2_11", data_format=IMAGE_ORDERING))(f2)
    pool211_2 = (Conv2DTranspose( nClasses , kernel_size=(2,2) ,  strides=(2,2) , use_bias=False, data_format=IMAGE_ORDERING ))(pool211)

    pool111 = ( Conv2D( nClasses , ( 1 , 1 ) , activation='relu' , padding='same', name="pool1_11", data_format=IMAGE_ORDERING))(f1)

    o = Add(name="add")([pool411_2, pool311_2, conv7_4, pool211_2, pool111])
    o = Conv2DTranspose( nClasses , kernel_size=(2,2) ,  strides=(2,2) , use_bias=False, data_format=IMAGE_ORDERING )(o)
    o = (Activation('softmax'))(o)
    
    model = Model(img_input, o)

    #######################################

    return model


def DSC(Yi,y_predi):
    # the function computes and prints out the Dice similarity. 
    # Yi is the ground truth and y_predi is the predicted lable for each pixel

    TP = np.sum( (Yi == 1)&(y_predi==1) )
    FP = np.sum( (Yi != 1)&(y_predi==1) )
    FN = np.sum( (Yi == 1)&(y_predi != 1)) 
    DSC = (2*TP)/(2*TP+FP+FN)
    print("DSC: {:4.3f}".format(DSC))

def post_processing(pred):
    # the function process the segmented images using opening and closing morphological operations. 
    # to remove holes and small objects regions assuming these area do not belong to an object.
	post_pred = np.zeros(shape = pred.shape)
	post_pred = np.copy(pred)
	for i in range(pred.shape[0]):
		post_pred[i,:,:] = scipy.ndimage.binary_opening((pred[i,:,:]), structure=np.array([[0,1,0],[1,1,1],[0,1,0,]])).astype(np.int)
		tmp = scipy.ndimage.binary_closing((pred[i,:,:]), structure=np.array([[0,1,0],[1,1,1],[0,1,0,]])).astype(np.int)
		tmp = scipy.ndimage.binary_opening(tmp, structure=np.array([[0,1,0],[1,1,1],[0,1,0,]])).astype(np.int)
		post_pred[i,1:-1,1:-1] = tmp[1:-1,1:-1]

	return post_pred

if __name__ == "__main__":

    dir_data = "crater_data/slices/train/"   # Directory where the data is saved
    n_classes= 2    # number of classes in the image, 2 as foreground and background
    input_height , input_width = 224 , 224
    
    # make the train set. 
    make_train_set('tile1_24')
    make_train_set('tile1_25')
    make_train_set('tile2_24')
    make_train_set('tile2_25')
    
    X_add, Y_add = extract_data_adds(dir_data)  # Extract two list of addresses: images and masks addresses.
    
    X, Y = extract_data(X_add,Y_add, input_width, input_height)    # two arrays of images and labels are extracted.
    
    output_height = input_height
    output_width = input_width
    
    model = FCN8_custom(n_classes,  
                 input_height = input_height, 
                 input_width  = input_width)
    
    print(model.summary())
    
    train_rate = 0.8    # what percentage of the files in the train folder serves as the training sample. the rest will be used as validation samples.
    index_train = np.random.choice(X.shape[0],int(X.shape[0]*train_rate),replace=False)   # in case we want to select the train samples randomly from the list.
    
    index_test  = list(set(range(X.shape[0])) - set(index_train))
                                
    X_train, y_train = X[index_train],Y[index_train]
    X_test, y_test = X[index_test],Y[index_test]
    
    sgd = optimizers.SGD(lr=1E-1, decay=5**(-4), momentum=0.9, nesterov=True)
    # sgd = optimizers.adam(lr=1E-4, decay=5**(-4))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    
    # model checkpoints
    checkpoint = ModelCheckpoint('models/seg-fcn-model.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    
    hist1 = model.fit(X_train,y_train, validation_split=0.33, epochs=150, batch_size=32, callbacks=callbacks_list, verbose=2)
    
    #hist1 = model.fit(X_train,y_train,
    #                  validation_data=(X_test,y_test),
    #                  batch_size=32,epochs=100,verbose=2) #change epochs to 1000
    
    # serialize model to JSON
    model_json = model.to_json()
    with open("models/seg-fcn-model.json", "w") as json_file:             
        json_file.write(model_json) 

    # serialize weights to HDF5
    model.save_weights("models/seg-fcn-model.h5")
    print("Saved model to disk")
    
    model.save('models/seg-fcn-model')
    print('model saved.')
    
    for key in ['loss', 'val_loss']:
        plt.plot(hist1.history[key],label=key)
    plt.legend()
    plt.savefig('results/loss_history.png', bbox_inches='tight', dpi=400)
    #plt.show()
    plt.figure()
    plt.close('all')
    #input('history')
    
    y_pred = model.predict(X_test)  # predicted mask for all samples in test set. shape = sample_no x height x width x 2
    y_predi = np.argmax(y_pred, axis=3)     # predicted class number of every pixel in the image, shape = sample_no x height x width
    y_testi = np.argmax(y_test, axis=3)     # class number of every pixel in the image, shape = sample_no x height x width
    
    y_predi_post = post_processing(y_predi)
    DSC(y_testi,y_predi_post)
    
    # the next for loop shows the testing image, ground truth and segmented area.
    for i in range(X_test.shape[0]):
        img_is  = (X_test[i] + 1)*(255.0/2)
        seg = y_predi[i]
        segtest = y_testi[i]
    
        fig = plt.figure(figsize=(10,30))    
        ax = fig.add_subplot(2,2,1)
        ax.imshow(img_is/255.0)
        ax.set_title("original")
        
        ax = fig.add_subplot(2,2,2)
        ax.imshow(give_color_to_seg_img(seg,n_classes))
        ax.set_title("predicted class")
        
        ax = fig.add_subplot(2,2,3)
        ax.imshow(give_color_to_seg_img(segtest,n_classes))
        ax.set_title("true class")
    
        ax = fig.add_subplot(2,2,4)
        ax.imshow(give_color_to_seg_img(y_predi_post[i],n_classes))
        ax.set_title("predicted after post processing")
        plt.savefig('results/training_sample_output_' + str(i) + '.png', bbox_inches='tight', dpi=400)
        #plt.close('all')
        plt.show()
    
    #input('visualize performance')
    
    
    print("end of main")
    
    