import torch
from ultralytics import YOLO
import numpy as np
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class KeypointsModel:
    def __init__(self, model_file_path: str) -> object:
        try:
            self.model = YOLO(model_file_path)
            print('Pose model loaded')
        except:
            print('There is no .pt file')
        return None
    def inference(self, img:Image) -> dict:
        try:
            self.results = self.model.predict(source=img, conf=0.4, device=DEVICE, save=False, save_txt=False, verbose=False)[0]
            d = {
                'classes':np.array(self.results.boxes.cls.cpu().numpy(),dtype=np.int8),
                'bboxes':np.array(self.results.boxes.xyxy.cpu().numpy(),dtype=np.float32),
                'keypoints':np.array(self.results.keypoints.xy.cpu().numpy(),dtype=np.float32)
            }
            return d
        except:
            print('key except')
            return None
    def get_results(self):
        return self.results
    def get_classes(self):
        return self.results.boxes.cls
    def get_bboxes(self):
        return self.results.boxes.xywh
    def get_normalized_bboxes(self):
        return self.results.boxes.xywhn
    def get_keypoints(self):
        return self.results.keypoints.xy
    def get_normalized_keypoints(self):
        return self.results.keypoints.xyn
    def get_dictionary(self):
        d = {
            'classes':np.array(self.results.boxes.cls,dtype=np.int8),
            'bboxes':np.array(self.results.boxes.xyxy,dtype=np.float32),
            'keypoints':np.array(self.results.keypoints.xy,dtype=np.float32)
        }
        return d

