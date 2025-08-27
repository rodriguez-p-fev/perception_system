import os
import time
import torch
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


CUDA =  torch.cuda.is_available()
print(CUDA)

perspective_transformer = PerspectiveTransformer.PerspectiveTransformer(calibration.wayside["SOURCE"], calibration.wayside["DEST"])
segmentation_inferences = SegmentationInference.SegmentationInference(config.segmentation_model_path, load_files.segmentation_model_files)
bboxes_inferences = BboxesInference.BboxesInference(config.keypoints_model_path, load_files.keypoints_model_files)

#for img_idx in range(len(load_files.imgs_files)):
#for img_idx in range(75,90):
if(True):
    img_idx=0
    
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
    #segments_set.set_active_path(start_node)




    # ******** LOCAL CODE DRAW AND SAVE IMAGE *************************************************************************************************************
    seg_img = img_array
    indexs = [start_node,0,6,1,5]
    for idx in indexs:
        seg = segments_set.get_segments()[idx]
        #if(seg.get_class() != 0):
        if True:
            rnd_color = (255,0,0)
            for p in seg.get_polygons():
                seg_img = draw.draw_polygon(seg_img, p.get_polygon(),rnd_color)
            seg_img = draw.draw_bbox(seg_img, seg.get_bbox(),rnd_color, text_bbox=Segment.segment_classes[seg.get_class()])
    seg_img = Image.fromarray(seg_img)
    seg_img.save(output_file)
