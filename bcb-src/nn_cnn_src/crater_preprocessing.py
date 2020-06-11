import cv2
import glob
import os
import imutils

def preprocess(tile_img, dataset_type, img_dimensions=(50, 50)):
    src = os.path.join('crater_data', 'samples', dataset_type, tile_img)
    dst = os.path.join('crater_data', 'samples', dataset_type, 'normalized_images')
    tgt_height, tgt_width = img_dimensions

    # create new directories if necessary
    for imgtype in ['crater', 'non-crater']:
        tgdir = os.path.join(dst, imgtype)
        if not os.path.isdir(tgdir):
            os.makedirs(tgdir)
            
    for src_filename in glob.glob(os.path.join(src, '*', '*.png')):

        pathinfo = src_filename.split(os.path.sep)
        img_type = pathinfo[-2] # crater or non-crater
        filename = pathinfo[-1] # the actual name of the jpg

        dst_filename = os.path.join(dst, img_type, filename)

        # read the original image and get size info
        src_img = cv2.imread(src_filename)

        # resize image, normalize and write to disk
        scaled_img = cv2.resize(src_img, (tgt_height, tgt_width))
        
        cv2.normalize(scaled_img, scaled_img, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(dst_filename, scaled_img)
    
    print(tile_img + " Done! ")

# in our dataset the ratio of positive to negative samples are 1 to 4 
# This script rotates only positive samples by 90, 180 and 270 degrees and save
# the samples to ratio became 1 to 1.
def positive_rotation_preprocess(tileimg, img_dimensions=(50, 50)):
    src = os.path.join('crater_data', 'images', tileimg)
    dst = os.path.join('crater_data', 'images', 'normalized_images')
    tgt_height, tgt_width = img_dimensions

    # create new directories if necessary
    for imgtype in ['crater', 'non-crater']:
        tgdir = os.path.join(dst, imgtype)
        if not os.path.isdir(tgdir):
            os.makedirs(tgdir)

    for src_filename in glob.glob(os.path.join(src, '*', '*.png')):
        #print(src_filename)
        pathinfo = src_filename.split(os.path.sep)
        img_type = pathinfo[-2] # crater or non-crater
        filename = pathinfo[-1] # the actual name of the jpg

        # read the original image and get size info
        src_img = cv2.imread(src_filename)
        scaled_img = cv2.resize(src_img, (tgt_height, tgt_width))
        
        if img_type == 'crater':
            
            rotated90 = imutils.rotate_bound(scaled_img, 90)
            rotated180 = imutils.rotate_bound(scaled_img, 180)
            rotated270 = imutils.rotate_bound(scaled_img, 270)
            
            dst_filename90 = os.path.join(dst, img_type, '90_'+filename)
            dst_filename180 = os.path.join(dst, img_type, '180_'+filename)
            dst_filename270 = os.path.join(dst, img_type, '270_'+filename)
    
            cv2.normalize(rotated90, rotated90, 0, 255, cv2.NORM_MINMAX)
            cv2.normalize(rotated180, rotated180, 0, 255, cv2.NORM_MINMAX)
            cv2.normalize(rotated270, rotated270, 0, 255, cv2.NORM_MINMAX)
            
            cv2.imwrite(dst_filename90, rotated90)
            cv2.imwrite(dst_filename180, rotated180)
            cv2.imwrite(dst_filename270, rotated270)
            
        else:
            
            dst_filename = os.path.join(dst, img_type, filename)
            cv2.normalize(scaled_img, scaled_img, 0, 255, cv2.NORM_MINMAX)
            cv2.imwrite(dst_filename, scaled_img)
    
    print("Done!")