import os
import torch
from PIL import Image

import config
from inference_files import load_files
from inference_files import SegmentationInference
from inference import inferences
from calibration import PerspectiveTransformer
from calibration import calibration
from track_detection import SegmentsSet
from wayside import wayside_functions
from draw import draw

CUDA =  torch.cuda.is_available()
print(CUDA)

perspective_transformer = PerspectiveTransformer.PerspectiveTransformer(calibration.wayside["SOURCE"], calibration.wayside["DEST"])
segmentation_inferences = SegmentationInference.SegmentationInference(config.segmentation_model_path, load_files.segmentation_model_files)

for img_idx in range(len(load_files.imgs_files)):
#for img_idx in range(75,90):
#if(True):
#    img_idx=0
    
    if(CUDA):
        # ******** ROS NODES ***********************************************************************************************
        img_array = load_files.get_image(img_idx)
        segmentation_dict = inferences.segmentation_model.inference(img_array)
    else:
        # ******** LOCAL CODE ***********************************************************************************************
        img_array, segmentation_dict = load_files.get_image_and_segmentation_dictionary(img_idx, segmentation_inferences)
    output_file = os.path.join(config.output_path, load_files.imgs_files[img_idx])


    # ******** TRACK DETECTION NODE ***********************************************************************************************
    #CREATE OBJECTS FROM INFERENCES
    if(segmentation_dict is not None):
        segments_set = SegmentsSet.SegmentsSet(segmentation_dict, perspective_transformer)
        selected_node = segments_set.get_closer_node(calibration.wayside["SOURCE"][0])

        selected_track  = segments_set.get_segments()[selected_node].get_polygons()[0].get_polygon()
        left_fvl, right_fvl, left_rfw, right_rfw = wayside_functions.get_fv_row_wayside_lines(selected_track, 
                             perspective_transformer, 
                             img_array.shape)
    else:
        left_fvl, right_fvl, left_rfw, right_rfw = [],[],[],[]

    # ******** LOCAL CODE DRAW AND SAVE IMAGE *************************************************************************************************************
    seg_img = img_array
    seg_img = draw.draw_polygon(seg_img, selected_track, [0,255,0])
    seg_img = Image.fromarray(seg_img)
    seg_img = draw.draw_foul_volume_lines(seg_img, [left_fvl,right_fvl], line_color="red")
    seg_img = draw.draw_foul_volume_lines(seg_img, [left_rfw,right_rfw], line_color="yellow")
    seg_img.save(output_file)

