# import the necessary packages
import numpy as np
import pandas as pd

# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
 
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
 
	# initialize the list of picked indexes	
	pick = []
 
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
 
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
 
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
 
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
 
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
 
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
 
	# return only the bounding boxes that were picked using the
	# integer data type
	return pick, boxes[pick].astype("int")

# this function gets crater data (x,y,s,p) as dataframe and calls the non_max_suppression_fast
def NMS(Crater_data):
    
    x = Crater_data[0].values.tolist()
    y = Crater_data[1].values.tolist()
    s = Crater_data[2].values.tolist()
    p = Crater_data[3].values.tolist()
    
    x1 = np.subtract(x ,s)
    y1 = np.subtract(y, s)
    x2 = np.add(x, s)
    y2 = np.add(y, s)
    
    boxes = np.array(zip(x1,y1,x2,y2))
    
    pick, pick_box = non_max_suppression_fast(boxes, 0.5)
    
    pick_circle = []
    for i in pick:
        x_center = x[i]
        y_center = y[i]
        size = s[i]
        prob = p[i]
        pick_circle += [[x_center,y_center,size, prob]]

    return pd.DataFrame(pick_circle)
    
    
    
    
    
    
    
    