import os
import numpy as np
from PIL import Image
import config


imgs_files               = sorted(os.listdir(config.imgs_path))
segmentation_model_files = sorted(os.listdir(config.segmentation_model_path))
keypoints_model_files    = sorted(os.listdir(config.keypoints_model_path))


def get_image(img_idx:int):
    img_file_path                = os.path.join(config.imgs_path, imgs_files[img_idx])
    img = Image.open(img_file_path).convert("RGB")
    img_array = np.array(img)
    return img_array
def get_image_and_models_dictionaries(img_idx:int, segmentation_inferences:object, bboxes_inferences:object):
    img_file_path                = os.path.join(config.imgs_path, imgs_files[img_idx])
    img = Image.open(img_file_path).convert("RGB")
    img_array = np.array(img)
    print_info(img_idx, img.size)
    #GET MODELS INFERENCES
    segmentation_dict = segmentation_inferences.get_inference(img_idx, img.size)
    keypointsbboxes_dict = bboxes_inferences.get_inference(img_idx, img.size)
    return img_array, segmentation_dict, keypointsbboxes_dict
def get_image_and_segmentation_dictionary(img_idx:int, segmentation_inferences:object):
    img_file_path                = os.path.join(config.imgs_path, imgs_files[img_idx])
    img = Image.open(img_file_path).convert("RGB")
    print_info(img_idx, img.size)
    img_array = np.array(img)
    #GET MODELS INFERENCES
    segmentation_dict = segmentation_inferences.get_inference(img_idx, img.size)
    return img_array, segmentation_dict
def print_info(img_idx, img_dim):
    print('**********************************')
    print(f'Image idx: {img_idx}')
    print(f'Image name: {imgs_files[img_idx]}')
    print(f'image shape: {img_dim}')