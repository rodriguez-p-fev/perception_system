import os
import torch
import numpy as np
from PIL import Image
import config
from inference_files import load_files
from inference_files import SegmentationInference
from calibration import PerspectiveTransformer
from calibration import calibration
from track_detection import SegmentsSet
from wayside import wayside_functions
from draw import draw
from side_lines import side_lines


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
        print("CUDA")
        from inference import inferences
        img_array = load_files.get_image(img_idx)
        segmentation_dict = inferences.segmentation_model.inference(img_array)
    else:
        # ******** LOCAL CODE ***********************************************************************************************
        print("CPU")
        img_array, segmentation_dict = load_files.get_image_and_segmentation_dictionary(img_idx, segmentation_inferences)
    output_file = os.path.join(config.output_path, load_files.imgs_files[img_idx])

    # ******** TRACK DETECTION NODE ***********************************************************************************************
    #CREATE OBJECTS FROM INFERENCES
    try:
        segments_set = SegmentsSet.SegmentsSet(segmentation_dict, perspective_transformer)

        #COMBINE BOTH MODELS INFERENCES INTO ONE OBJECT
        segments_set.initialize_segments([])

        selected_node = segments_set.get_closer_node(calibration.wayside["SOURCE"][1])

        active_polygons  = segments_set.get_segments()[selected_node].get_polygons()[0].get_polygon()
        #left_fvl, right_fvl, left_rfw, right_rfw = wayside_functions.get_fv_row_wayside_lines(selected_track, 
        #                     perspective_transformer, 
        #                     img_array.shape)
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
    
        left_points_perspective, right_points_perspective = wayside_functions.filter_vertical_polygon(selected_track_perspective, error_margin=0.8)
        
        left_curve_perspective = side_lines.curve_fitting(left_points_perspective, n_points=500, grad=1)
        right_curve_perspective = side_lines.curve_fitting(right_points_perspective, n_points=500, grad=1)
    
        left_fvl_perspective, right_fvl_perspective, left_rfw_perspective, right_rfw_perspective = wayside_functions.get_fv_row_lines(left_curve_perspective, right_curve_perspective)
        #left_points = perspective_transformer.inverse_transform(left_curve_perspective)
        #right_points = perspective_transformer.inverse_transform(right_curve_perspective)#

        left_curve = perspective_transformer.inverse_transform(left_curve_perspective)
        right_curve = perspective_transformer.inverse_transform(right_curve_perspective)
        left_fvl = perspective_transformer.inverse_transform(left_fvl_perspective)
        right_fvl = perspective_transformer.inverse_transform(right_fvl_perspective)
        left_rfw = perspective_transformer.inverse_transform(left_rfw_perspective)
        right_rfw = perspective_transformer.inverse_transform(right_rfw_perspective)
    except:
        print("error in foul volume lines")
        left_fvl, right_fvl, left_rfw, right_rfw = [],[],[],[]

    # ******** LOCAL CODE DRAW AND SAVE IMAGE *************************************************************************************************************
    import cv2
    import matplotlib.pyplot as plt

    seg_img = img_array
    seg_img = draw.draw_polygon(seg_img, active_polygons, [0,255,0])
    seg_img = Image.fromarray(seg_img)
    seg_img = draw.draw_foul_volume_lines(seg_img, [left_curve, right_curve], line_color="green")
    seg_img = draw.draw_foul_volume_lines(seg_img, [left_fvl, right_fvl], line_color="red")
    seg_img = draw.draw_foul_volume_lines(seg_img, [left_rfw, right_rfw], line_color="yellow")
    seg_img.save(output_file)

    points = np.array(calibration.wayside["SOURCE"])
    plt.scatter(selected_track_perspective[:,0],selected_track_perspective[:,1],s=2)
    seg_img = Image.fromarray(img_array)
    #plt.imshow(seg_img)
    #plt.scatter(points[:,0],points[:,1],s=10)
    #plt.xlim(-15*GAUGE, 20*GAUGE)
    #plt.ylim(500000, -4000000)
    plt.savefig("./plot_test.png")
    plt.cla()

