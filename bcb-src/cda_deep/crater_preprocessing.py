import cv2
import glob
import os

def preprocess(tile_name, dest_folder, img_dimensions=(50, 50)):
    src = os.path.join('crater_data', 'images', tile_name)
    dst = os.path.join('crater_data', dest_folder)
    tgt_height, tgt_width = img_dimensions

    # create new directories if necessary
    for imgtype in ['crater', 'non-crater']:
        tgdir = os.path.join(dst, imgtype)
        if not os.path.isdir(tgdir):
            os.makedirs(tgdir)

    for src_filename in glob.glob(os.path.join(src, '*', '*.jpg')):
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
    
    print("Done!")
