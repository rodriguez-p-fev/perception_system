import numpy as np
from track_detection import Segment
from side_lines import side_lines
from calibration import PerspectiveTransformer


def filter_polygon_horizontal(selected_points: np.ndarray, error_margin = 0.8, GAUGE=1435):
    selected_points = np.array(selected_points)
    left = []
    right = []
    bottom = selected_points[:,1].min()
    top = selected_points[:,1].max()
    step = (top-bottom)/1000
    bottom = bottom - step/2
    top = top + step/2

    for i in range(int(bottom + step),int(top-step),int(step)):
        filtered = np.where((selected_points[:,1]>i-step) & (selected_points[:,1]<i+step))[0]
        if(len(filtered)>0):
            l = selected_points[filtered][:,0].min()
            r = selected_points[filtered][:,0].max()
            diff = r-l
            if(diff>GAUGE*(1-error_margin) and diff<GAUGE*(1+error_margin)):
                l_idx = np.where(selected_points[filtered][:,0] == l)[0][0]
                r_idx = np.where(selected_points[filtered][:,0] == r)[0][0]
                left.append(selected_points[filtered][l_idx])
                right.append(selected_points[filtered][r_idx])
    left = np.array(left)
    right = np.array(right)
    left_curve = side_lines.curve_fitting(left, n_points=500, grad=4)
    right_curve = side_lines.curve_fitting(right, n_points=500, grad=4)
    return left_curve, right_curve
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



def get_fv_row_wayside_lines(shape: np.ndarray, 
                             perspective_transformer: PerspectiveTransformer, 
                             img_shape:np.ndarray):
    selected_track_perspective = perspective_transformer.transform(shape)
    left_curve_perspective, right_curve_perspective = filter_polygon_horizontal(selected_track_perspective)

    left_fvl_perspective, right_fvl_perspective, left_rfw_perspective, right_rfw_perspective = get_fv_row_lines(left_curve_perspective, right_curve_perspective)
    
    #left_curve = perspective_transformer.inverse_transform(left_curve_perspective)
    #right_curve = perspective_transformer.inverse_transform(right_curve_perspective)
    left_fvl = perspective_transformer.inverse_transform(left_fvl_perspective)
    right_fvl = perspective_transformer.inverse_transform(right_fvl_perspective)
    left_rfw = perspective_transformer.inverse_transform(left_rfw_perspective)
    right_rfw = perspective_transformer.inverse_transform(right_rfw_perspective)

    return left_fvl, right_fvl, left_rfw, right_rfw
