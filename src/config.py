from os.path import isfile
import os
from configparser import ConfigParser
from cli import args
print(args.main_path)                 
if not isfile(os.path.join(os.getcwd(), 'config.ini')):
    print("config file doesn't exist")
else:
    cfgParser = ConfigParser()
    cfgParser.read(os.path.join(os.getcwd(), 'config.ini'))
    sections = cfgParser.sections()
    imgs_path               = os.path.join(args.main_path, 'images')
    segmentation_model_path = os.path.join(args.main_path, 'segmentations')
    keypoints_model_path    = os.path.join(args.main_path, 'keypoints_detections')
    output_path             = os.path.join(args.main_path, 'output')
