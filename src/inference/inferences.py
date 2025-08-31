from ultralytics import YOLO
import os
import numpy as np
from inference import SegmentationModel
from inference import KeypointsModel
from PIL import Image
import config

imgs_files = sorted(os.listdir(config.imgs_path))

directory = os.path.dirname(os.path.abspath(__file__))
seg_model_path = os.path.join(directory, 'models/segmentation_model.pt')
keypts_model_path = os.path.join(directory, 'models/keypoint_model.pt')

segmentation_model = SegmentationModel.SegmentationModel(seg_model_path)
keypoints_model = KeypointsModel.KeypointsModel(keypts_model_path)
