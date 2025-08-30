import os
import time
import torch
import numpy as np
from PIL import Image

import config
from inference_files import load_files
from inference_files import SegmentationInference
from inference_files import BboxesInference
from inference import inferences
from calibration import PerspectiveTransformer
from calibration import calibration
from track_detection import Segment
from track_detection import SegmentsSet
from track_detection import KeypointsBboxesSet
from draw import draw
from wayside import wayside_functions
from side_lines import side_lines

CUDA =  torch.cuda.is_available()
print(CUDA)

perspective_transformer = PerspectiveTransformer.PerspectiveTransformer(calibration.normal["SOURCE"], calibration.normal["DEST"])
segmentation_inferences = SegmentationInference.SegmentationInference(config.segmentation_model_path, load_files.segmentation_model_files)
bboxes_inferences = BboxesInference.BboxesInference(config.keypoints_model_path, load_files.keypoints_model_files)



for img_idx in range(len(load_files.imgs_files)):
    if(CUDA):
        print("CUDA")
        # ******** ROS NODES ***********************************************************************************************
        img_array = load_files.get_image(img_idx)
        segmentation_dict = inferences.segmentation_model.inference(img_array)
        keypointsbboxes_dict = inferences.keypoints_model.inference(img_array)
    else:
        print("CPU")
        # ******** LOCAL CODE ***********************************************************************************************
        img_array, segmentation_dict, keypointsbboxes_dict = load_files.get_image_and_models_dictionaries(img_idx, segmentation_inferences, bboxes_inferences)
    output_file = os.path.join(config.output_path, load_files.imgs_files[img_idx])

    
    # ******** TRACK DETECTION NODE ***********************************************************************************************
    try:
        start = time.time()
        #CREATE OBJECTS FROM INFERENCES
        segments_set = SegmentsSet.SegmentsSet(segmentation_dict, perspective_transformer)
        keypointsbboxes_sets = KeypointsBboxesSet.KeypointsBboxesSet(keypointsbboxes_dict)
    
        #COMBINE BOTH MODELS INFERENCES INTO ONE OBJECT
        segments_set.initialize_segments(keypointsbboxes_sets.get_keypoints_bboxes())

        #SELECT START NODE
        start_node = segments_set.get_closer_node(calibration.normal["SOURCE"][0])
        print(f'start node: {start_node}')

        #CREATE THE GRAPH OF NODES
        segments_set.set_nodes_graph(calibration.normal["SOURCE"][0])

        #CREATE ACTIVE PATH, UNACTIVE PATH AND UNKNOWN SEGMENTS
        active_nodes =  segments_set.set_active_path(start_node)
        active_polygons =  segments_set.get_active_polygons(active_nodes)
        active_polygons = active_polygons
    except:
        print("Warning: active polygons empty")
        active_polygons = []
    # OUTPUT
    #       Active polygons: list of np.ndarray
    # ******** FOUL VOLUME AND RIGHT OF WAY LINES NODE ***********************************************************************************************
    try:
        active_polygons_points = []
        for p in active_polygons:
            active_polygons_points.extend(p)
        ravel_arr = np.ravel(active_polygons_points)
        active_points = np.reshape(np.ravel(ravel_arr),(int(len(ravel_arr)/2),2))

        selected_track_perspective = perspective_transformer.transform(active_points)
        selected_track_perspective = selected_track_perspective[np.where(selected_track_perspective[:,1] > 0)]
        selected_track_perspective = selected_track_perspective[np.where(selected_track_perspective[:,1] < 400000)]
    
        left_points_perspective, right_points_perspective = wayside_functions.filter_polygon_vertical(selected_track_perspective)
        left_curve_perspective = side_lines.curve_fitting(left_points_perspective, n_points=500, grad=4)
        right_curve_perspective = side_lines.curve_fitting(right_points_perspective, n_points=500, grad=4)
    

        left_fvl_perspective, right_fvl_perspective, left_rfw_perspective, right_rfw_perspective = wayside_functions.get_fv_row_lines(left_curve_perspective, right_curve_perspective)
        #left_points = perspective_transformer.inverse_transform(left_curve_perspective)
        #right_points = perspective_transformer.inverse_transform(right_curve_perspective)#

        #left_curve = perspective_transformer.inverse_transform(left_curve_perspective)
        #right_curve = perspective_transformer.inverse_transform(right_curve_perspective)
        left_fvl = perspective_transformer.inverse_transform(left_fvl_perspective)
        right_fvl = perspective_transformer.inverse_transform(right_fvl_perspective)
        left_rfw = perspective_transformer.inverse_transform(left_rfw_perspective)
        right_rfw = perspective_transformer.inverse_transform(right_rfw_perspective)
    except:
        left_fvl, right_fvl, left_rfw, right_rfw = [],[],[],[]
    # ******** LOCAL CODE DRAW AND SAVE IMAGE *************************************************************************************************************
    seg_img = img_array
    #active_polygons =active_polygons[:6]
    for p in active_polygons:
        rnd_color = (50,255,50)
        seg_img = draw.draw_polygon(seg_img, p, rnd_color)
    seg_img = Image.fromarray(seg_img)
    seg_img = draw.draw_foul_volume_lines(seg_img, [left_fvl, right_fvl], line_color="red")
    seg_img = draw.draw_foul_volume_lines(seg_img, [left_rfw, right_rfw], line_color="yellow")
    seg_img.save(output_file)