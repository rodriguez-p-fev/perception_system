import numpy as np
import cv2

def get_polygon_mask(polygons, img_shape):
    blank = np.zeros(shape=(img_shape[0],img_shape[1]),dtype=np.float32)
    for shape in polygons:
        points = np.array(shape,np.int32)
        cv2.fillPoly(blank,[points],255)
    blank = np.expand_dims(blank,axis=2)
    mask_array = np.array(blank)
    mask_array = np.uint8(mask_array)
    mask_array = mask_array.reshape(mask_array.shape[0],mask_array.shape[1])
    return mask_array
def get_polygon_edges(polygons: np.ndarray, img_shape):
    mask_array = get_polygon_mask(polygons, img_shape)
    edges = cv2.Canny(mask_array,100,200)
    y,x = np.where(edges!=0)
    output = np.array([x,y])
    output = np.reshape(output, (2,len(output[0]))).T
    return output
def get_contour(pointset, step):
    left = []
    right = []
    if(len(pointset) > 0):
        y_start = int(pointset[:,1].max())
        y_end = int(pointset[:,1].min())
        #step = (y_start - y_end)/n_steps
        idxs = np.where(pointset[:,1] == y_start)
        left.append([pointset[idxs][:,0].min(),y_start])
        right.append([pointset[idxs][:,0].max(),y_start])
        for i in range(int(y_start-5), int(y_end+5), -int(step)):
            idxs = np.where(pointset[:,1] == i)
            filtered_pointset = pointset[idxs]
            if(len(filtered_pointset) > 0):
                left.append([filtered_pointset[:,0].min(),i])
                right.append([filtered_pointset[:,0].max(),i])
        idxs = np.where(pointset[:,1] == y_end)
        if(len(filtered_pointset) > 0):
            left.append([filtered_pointset[:,0].min(),y_end])
            right.append([filtered_pointset[:,0].max(),y_end])
    return np.array(left), np.array(right)
def get_contour_aux(pointset, step):
    left = []
    right = []
    y_start = int(pointset[:,1].max())
    y_end = int(pointset[:,1].min())
    idxs = np.where((pointset[:,1] <= y_start+10) & (pointset[:,1] > y_start-10))
    left.append([pointset[idxs][:,0].min(),y_start])
    right.append([pointset[idxs][:,0].max(),y_start])
    for i in range(y_start-5, y_end+5, -step):
        #idxs = np.where(pointset[:,1] == i)
        idxs = np.where((pointset[:,1] <= i+10) & (pointset[:,1] > i-10))
        filtered_pointset = pointset[idxs]
        if(len(filtered_pointset) > 0):
            left.append([filtered_pointset[:,0].min(),i])
            right.append([filtered_pointset[:,0].max(),i])
    idxs = np.where(pointset[:,1] == y_end)
    left.append([filtered_pointset[:,0].min(),y_end])
    right.append([filtered_pointset[:,0].max(),y_end])
    return np.array(left), np.array(right)
def foul_volume_points(left_values, right_values):
    GAUGE = 1435
    ratio1 = (1828/GAUGE)-(1/2) #6ft = 1828mm
    ratio2 = (6096/GAUGE)-(1/2)  #20ft = 6096mm
    left_fvl = []
    right_fvl = []
    left_fvl2 = []
    right_fvl2 = []
    for i in range(len(left_values)):
        dist = right_values[i][0] - left_values[i][0]
        left_fvl.append([left_values[i][0] - (dist*ratio1), left_values[i][1]])
        right_fvl.append([right_values[i][0] + (dist*ratio1), right_values[i][1]])
        left_fvl2.append([left_values[i][0] - (dist*ratio2), left_values[i][1]])
        right_fvl2.append([right_values[i][0] + (dist*ratio2), right_values[i][1]])
    left_fvl = np.array(left_fvl)
    right_fvl = np.array(right_fvl)
    left_fvl2 = np.array(left_fvl2)
    right_fvl2 = np.array(right_fvl2)
    return left_fvl, right_fvl, left_fvl2, right_fvl2
def curve_fitting(rail_pointset, n_points=100, grad=4):
    coefficients = np.polyfit(rail_pointset[:,1],rail_pointset[:,0], grad)
    p = np.poly1d(coefficients)
    new_x = np.linspace(rail_pointset[:,1].min(),rail_pointset[:,1].max(), n_points)
    new_y = p(new_x)
    return np.concatenate((np.reshape(new_y, (len(new_y),1)), np.reshape(new_x, (1,len(new_x))).T), axis=1)
def get_contour_lines(active_path: np.ndarray, img_array: np.ndarray) -> list:
    if(len(active_path) > 0):
        edges_img = get_polygon_edges(img_array, active_path)
        left, right = get_contour(edges_img,10)
        left = curve_fitting(left)
        right = curve_fitting(right)
        return [left, right]
    else: 
        return [[],[]]
def get_foul_volume_lines(active_path, img_array):
    if(len(active_path) > 0):
        edges_img = get_polygon_edges(img_array.shape, active_path)
        left, right = get_contour(edges_img,10)
        left = curve_fitting(left)
        right = curve_fitting(right)
        left_fvl, right_fvl, left_fvl2, right_fvl2 = foul_volume_points(left, right)

        left_fvl = curve_fitting(left_fvl)
        right_fvl = curve_fitting(right_fvl)
        left_fvl2 = curve_fitting(left_fvl2)
        right_fvl2 = curve_fitting(right_fvl2)
    else:
        left_fvl = np.array([])
        right_fvl = np.array([])
        left_fvl2 = np.array([])
        right_fvl2 = np.array([])
    return left_fvl, right_fvl, left_fvl2, right_fvl2

def template_lines(pointset: np.ndarray):
    left_side_1  = pointset[0]
    center_1     = pointset[1]
    right_side_1 = pointset[2]
    left_side_2  = pointset[3]
    center_2     = pointset[4]
    right_side_2 = pointset[5]

    center_line = curve_fitting(np.array([center_1,center_2]), grad=1)
    left_fvl = curve_fitting(np.array([left_side_1,left_side_2]), grad=1)
    right_fvl = curve_fitting(np.array([right_side_1,right_side_2]), grad=1)
    return left_fvl, center_line, right_fvl
def get_fv_row_lines(left_curve, right_curve, GAUGE=1435):
    sixfeet = 1828
    twentyfeet = 6096
    sixfeet_from_center = sixfeet - GAUGE/2
    twentyfeet_from_center = twentyfeet - GAUGE/2
    left_fvl = np.copy(left_curve)
    right_fvl = np.copy(right_curve)
    left_rfw = np.copy(left_curve)
    right_rfw = np.copy(right_curve)
    left_fvl[:,0] = left_fvl[:,0] - sixfeet_from_center
    right_fvl[:,0] = right_fvl[:,0] + sixfeet_from_center
    left_rfw[:,0] = left_fvl[:,0] - twentyfeet_from_center
    right_rfw[:,0] = right_fvl[:,0] + twentyfeet_from_center
    return left_fvl, right_fvl, left_rfw, right_rfw


