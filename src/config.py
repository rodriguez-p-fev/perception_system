from os.path import isfile
import os
from configparser import ConfigParser

                         
if not isfile(os.path.join(os.getcwd(), 'config.ini')):
    print("config file doesn't exist")
else:
    cfgParser = ConfigParser()
    cfgParser.read(os.path.join(os.getcwd(), 'config.ini'))
    sections = cfgParser.sections()
    imgs_path               = cfgParser.get('config', 'imgs_path')
    segmentation_model_path = cfgParser.get('config', 'segmentation_model_path')
    keypoints_model_path    = cfgParser.get('config', 'keypoints_model_path')
    output_path             = cfgParser.get('config', 'output_path')
