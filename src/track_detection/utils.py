import math
import numpy as np
import cv2
WIDTH = 4096.0
HEIGHT = 3040.0
def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
def vertical_weighted_distance(p1, p2, weight=1.35):
    return math.sqrt((abs(p1[0]-p2[0]))**2 + (weight*abs(p1[1]-p2[1]))**2) #+ max(0, p2[1]-p1[1])
def get_average_distance(left_top,left_bottom,right_top,right_bottom):
    left_distance = distance(left_top,left_bottom)
    right_distance = distance(right_top,right_bottom)
    average_distance = (left_distance+right_distance)/2
    return average_distance
def rescale_pointset(pointset: np.ndarray) -> np.ndarray:
    rescaled = np.copy(pointset)
    #rescaled[:,0] = rescaled[:,0] - rescaled[:,0].min()
    #rescaled[:,1] = rescaled[:,1] - rescaled[:,1].min()
    #rescaled[:,0] = 10*WIDTH*rescaled[:,0] / rescaled[:,0].max()
    #rescaled[:,1] = 10*HEIGHT*rescaled[:,1] / rescaled[:,1].max()
    return rescaled
def get_polygon_edges(img, shapes):
    blank = np.zeros(shape=(img.shape[0],img.shape[1]),dtype=np.float32)
    for shape in shapes:
        points = np.array(shape,np.int32)
        cv2.fillPoly(blank,[points],255)
    blank = np.expand_dims(blank,axis=2)
    mask_array = np.array(blank)
    mask_array = np.uint8(mask_array)
    edges = cv2.Canny(mask_array,100,200)
    y,x = np.where(edges!=0)
    output = np.array([x,y])
    output = np.reshape(output, (2,len(output[0]))).T
    return output
def bboxes_intersection(bbox1,bbox2):
    xdiff = min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0]) 
    ydiff = min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1])
    if(xdiff<0 or ydiff< 0):
        area_intersection = 0
    else:
        area_intersection = xdiff*ydiff
    return area_intersection
def bboxes_IoU(bbox1,bbox2):
    xdiff = min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0]) 
    ydiff = min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1])
    if(xdiff<0 or ydiff< 0):
        IoU = 0
    else:
        area_intersection = xdiff*ydiff
        area_union = (bbox1[2] - bbox1[0])*(bbox1[3] - bbox1[1]) + (bbox2[2] - bbox2[0])*(bbox2[3] - bbox2[1]) - area_intersection
        IoU = 100*area_intersection/area_union
    return IoU