import os
import time
import torch
import numpy as np
from PIL import Image

import config
from inference_files import load_files
from inference_files import SegmentationInference
from inference_files import BboxesInference

from calibration import PerspectiveTransformer
from calibration import calibration
from track_detection import Segment
from track_detection import SegmentsSet
from track_detection import KeypointsBboxesSet
from draw import draw
from wayside import wayside_functions
from side_lines import side_lines
from cli import args

CUDA =  torch.cuda.is_available()

perspective_transformer = PerspectiveTransformer.PerspectiveTransformer(calibration.normal["SOURCE"], calibration.normal["DEST"])
segmentation_inferences = SegmentationInference.SegmentationInference(config.segmentation_model_path, load_files.segmentation_model_files)
bboxes_inferences = BboxesInference.BboxesInference(config.keypoints_model_path, load_files.keypoints_model_files)

prev_class = 0
prev_dir = 2
prev_bbox = [4096/4,3040-900,3*4096/4,3040]

for img_idx in range(len(load_files.imgs_files)):
    if(CUDA):
        print("CUDA")
        from inference import inferences
        # ******** ROS NODES ***********************************************************************************************
        img_array = load_files.get_image(img_idx)
        segmentation_dict = inferences.segmentation_model.inference(img_array)
        keypointsbboxes_dict = inferences.keypoints_model.inference(img_array)
        output_file = os.path.join(config.output_path, inferences.imgs_files[img_idx])
    else:
        print("CPU")
        # ******** LOCAL CODE ***********************************************************************************************
        img_array, segmentation_dict, keypointsbboxes_dict = load_files.get_image_and_models_dictionaries(img_idx, segmentation_inferences, bboxes_inferences)
    output_file = os.path.join(config.output_path, load_files.imgs_files[img_idx])

    try:
        #CREATE THE GRAPH OF NODES
        start = time.time()
        #CREATE OBJECTS FROM INFERENCES
        segments_set = SegmentsSet.SegmentsSet(segmentation_dict, perspective_transformer)
        keypointsbboxes_sets = KeypointsBboxesSet.KeypointsBboxesSet(keypointsbboxes_dict)
    
        #COMBINE BOTH MODELS INFERENCES INTO ONE OBJECT
        segments_set.initialize_segments(keypointsbboxes_sets.get_keypoints_bboxes())

        #SELECT START NODE
        start_node = segments_set.get_start_node(prev_bbox)
        segments_set.set_nodes_graph(start_node)

        if(
            (Segment.segment_classes[segments_set.get_segments()[start_node].get_class()] in ["fp_turnout","tp_turnout","common"] and 
            Segment.direction_dict[segments_set.get_segments()[start_node].get_direction()] == "unknown") or 
            (segments_set.get_segments()[start_node].get_class() == prev_class and 
            segments_set.get_segments()[start_node].get_direction() != prev_dir)):
                segments_set.get_segments()[start_node].set_direction(prev_dir)
        #CREATE ACTIVE PATH, UNACTIVE PATH AND UNKNOWN SEGMENTS
        active_nodes =  segments_set.set_active_path(start_node)
        active_polygons =  segments_set.get_active_polygons(active_nodes)
        prev_dir = segments_set.get_segments()[start_node].get_direction()
        prev_bbox = segments_set.get_segments()[start_node].get_active_bbox()
        prev_class = segments_set.get_segments()[start_node].get_class()
    except:
        print("Warning: active polygons empty")
        active_polygons = []
    


    seg_img = img_array
    for p in active_polygons:
        rnd_color = (50,255,50)
        seg_img = draw.draw_polygon(seg_img, p, rnd_color)
    seg_img = Image.fromarray(seg_img)
    seg_img.save(output_file)
    #seg_img.save("img.jpg")