import os
import numpy as np

keypoit_model = {
    0:'track',
    1:'turnout'
}
class BboxesInference:
    def __init__(self, model_files_path: str, output_files: list) -> object:
        self.model_files_path=model_files_path
        self.output_files = output_files
        return None
    def get_inference(self, idx: int, img_shape: np.ndarray) -> dict:
        try:
            inference_file_path = os.path.join(self.model_files_path, self.output_files[idx])
            keypoints_str = self.get_keypoints_from_path(inference_file_path)
            return self.parse_keypoints_str_line(keypoints_str, img_shape)
        except:
            return None
    def get_keypoints_from_path(self,txt_path: str) -> list:
        keypoints = []
        f = open(txt_path, "r")
        for x in f:
            c_array = x.split(' ')
            keypoints.append(c_array)
        return keypoints
    def parse_keypoints_str_line(self, keypoints_str: list, img_shape: np.ndarray) -> dict:
        bboxes_classes = []
        bboxes = []
        keypoints = []
        for keypoints_str_line in keypoints_str:
            object_class = int(keypoints_str_line[0])
            bbox = np.array(keypoints_str_line[1:5], dtype = np.float16)
            keypts = np.array(keypoints_str_line[5:], dtype=np.float16)
            keypts = np.multiply(np.reshape(keypts,(6,3),order='A'),np.array([img_shape[0],img_shape[1],255])).astype(np.int16)  
            bbox = np.multiply(bbox, np.array([img_shape[0],img_shape[1],img_shape[0],img_shape[1]])).astype(np.float32)
            bbox[0:2] = bbox[0:2]-bbox[2:]/2
            bbox[2:] = bbox[0:2]+bbox[2:]
            bboxes_classes.append(object_class)
            bboxes.append(bbox)
            keypoints.append(keypts)
        d = {
            'classes':np.array(bboxes_classes,dtype=np.int8),
            'bboxes':np.array(bboxes,dtype=np.float32),
            'keypoints':np.array(keypoints,dtype=np.float32)
        }
        return d
