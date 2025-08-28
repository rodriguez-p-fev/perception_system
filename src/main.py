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
    print("update")
#for img_idx in range(75,90):
#if(True):
    #img_idx=5
    
    if(CUDA):
        # ******** ROS NODES ***********************************************************************************************
        img_array = load_files.get_image(img_idx)
        segmentation_dict = inferences.segmentation_model.inference(img_array)
        keypointsbboxes_dict = inferences.keypoints_model.inference(img_array)
    else:
        # ******** LOCAL CODE ***********************************************************************************************
        img_array, segmentation_dict, keypointsbboxes_dict = load_files.get_image_and_models_dictionaries(img_idx, segmentation_inferences, bboxes_inferences)
    output_file = os.path.join(config.output_path, load_files.imgs_files[img_idx])

    
    # ******** TRACK DETECTION NODE ***********************************************************************************************
    if(segmentation_dict is not None):
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
        segments_set.set_nodes_graph(calibration.wayside["SOURCE"][0])

        #CREATE ACTIVE PATH, UNACTIVE PATH AND UNKNOWN SEGMENTS
        active_nodes =  segments_set.set_active_path(start_node)
        active_polygons =  segments_set.get_active_polygons(active_nodes)
        active_polygons = active_polygons
    else:
        active_polygons = []
    # OUTPUT
    #       Active polygons: np.ndarray
    
    # ******** FOUL VOLUME AND RIGHT OF WAY LINES NODE ***********************************************************************************************
    #selected_track_perspective = perspective_transformer.transform(active_polygons)
    #selected_track_perspective = selected_track_perspective[np.where(selected_track_perspective[:,1] > 0)]
    #selected_track_perspective = selected_track_perspective[np.where(selected_track_perspective[:,1] < 300000)]
    #left_points_perspective, right_points_perspective = wayside_functions.filter_polygon_vertical(selected_track_perspective)
    #left_curve_perspective = side_lines.curve_fitting(left_points_perspective, n_points=500, grad=4)
    #right_curve_perspective = side_lines.curve_fitting(right_points_perspective, n_points=500, grad=4)
   # 
   # left_fvl_perspective, right_fvl_perspective, left_rfw_perspective, right_rfw_perspective = wayside_functions.get_fv_row_lines(left_curve_perspective, right_curve_perspective)
   # left_points = perspective_transformer.inverse_transform(left_curve_perspective)
   # right_points = perspective_transformer.inverse_transform(right_curve_perspective)#

    #left_curve = perspective_transformer.inverse_transform(left_curve_perspective)
    #right_curve = perspective_transformer.inverse_transform(right_curve_perspective)
    #left_fvl = perspective_transformer.inverse_transform(left_fvl_perspective)
    #right_fvl = perspective_transformer.inverse_transform(right_fvl_perspective)
    ##left_rfw = perspective_transformer.inverse_transform(left_rfw_perspective)
    #right_rfw = perspective_transformer.inverse_transform(right_rfw_perspective)
    #left_points = perspective_transformer.inverse_transform(left_points_perspective)
    #right_points = perspective_transformer.inverse_transform(right_points_perspective)
    # ******** LOCAL CODE DRAW AND SAVE IMAGE *************************************************************************************************************
    seg_img = img_array
    indexs = active_nodes[:]
    for idx in indexs:
        seg = segments_set.get_segments()[idx]
        rnd_color = (50,255,50)
        if(seg.get_class() == 1):
            if(seg.get_direction() == -1):
                seg_img = draw.draw_polygon(seg_img, seg.get_polygons()[0].get_polygon(),rnd_color)
                seg_img = draw.draw_polygon(seg_img, seg.get_polygons()[1].get_polygon(),rnd_color)
            else:
                seg_img = draw.draw_polygon(seg_img, seg.get_polygons()[0].get_polygon(),rnd_color)
                seg_img = draw.draw_polygon(seg_img, seg.get_polygons()[2].get_polygon(),rnd_color)
            #seg_img = draw.draw_bbox(seg_img, seg.get_bbox(),rnd_color, text_bbox=Segment.segment_classes[seg.get_class()])
        else:
            seg_img = draw.draw_polygon(seg_img, seg.get_polygons()[0].get_polygon(),rnd_color)
    seg_img = Image.fromarray(seg_img)
    #print(left_points_perspective)
    #seg_img = draw.draw_pointset(seg_img, np.array(left_points))
    #seg_img = draw.draw_pointset(seg_img, np.array(right_points))
    #seg_img = draw.draw_foul_volume_lines(seg_img, [left_curve, right_curve], line_color="green")
    #seg_img = draw.draw_foul_volume_lines(seg_img, [left_fvl, right_fvl], line_color="red")
    #seg_img = draw.draw_foul_volume_lines(seg_img, [left_rfw, right_rfw], line_color="yellow")
    #seg_img.save(output_file)
    
    seg_img = img_array
    for p in active_polygons:
        seg_img = draw.draw_polygon(seg_img, p, rnd_color)
    seg_img = Image.fromarray(seg_img)
    seg_img.save(output_file)
"""
    import matplotlib.pyplot as plt
    points = np.array(calibration.normal["SOURCE"])
    plt.scatter(active_polygons[:,0],active_polygons[:,1],s=2)
    seg_img = Image.fromarray(img_array)
    plt.imshow(seg_img)
    plt.scatter(points[:,0],points[:,1],s=10)
    #plt.xlim(-15*GAUGE, 20*GAUGE)
    #plt.ylim(500000, -4000000)
    plt.savefig("./plot_test")
    plt.cla()
    
    plt.scatter(selected_track_perspective[:,0],selected_track_perspective[:,1],s=2)
    plt.scatter(left_curve_perspective[:,0],left_curve_perspective[:,1],s=2)
    plt.scatter(right_curve_perspective[:,0],right_curve_perspective[:,1],s=2)
    #plt.xlim(-15*GAUGE, 20*GAUGE)
    #plt.ylim(500000, -4000000)
    plt.savefig("./per_test")
    plt.cla()

    plt.scatter(active_polygons[:,0],active_polygons[:,1],s=2)
    seg_img = Image.fromarray(img_array)
    plt.imshow(seg_img)
    plt.scatter(left_curve[:,0],left_curve[:,1],s=3)
    plt.scatter(right_curve[:,0],right_curve[:,1],s=3)

    plt.scatter(left_fvl[:,0],left_fvl[:,1],s=3)
    plt.scatter(right_fvl[:,0],right_fvl[:,1],s=3)
    plt.scatter(left_fvl[:,0],left_fvl[:,1],s=3)
    plt.scatter(left_rfw[:,0],left_rfw[:,1],s=3)
    plt.scatter(right_rfw[:,0],right_rfw[:,1],s=3)
    plt.savefig("./per_test_22")
    plt.cla()
"""