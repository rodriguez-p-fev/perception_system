import torch
from ultralytics import YOLO
import numpy as np
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class SegmentationModel:
    def __init__(self, model_file_path: str) -> object:
        try:
            self.model = YOLO(model_file_path)
            print('Segmentation model loaded')
        except:
            print('There is no .pt file')
        return None
    def inference(self, img:Image) -> dict:
        try:
            self.results = self.model.predict(source=img, conf=0.4, device=DEVICE, save=False, save_txt=False, verbose=False)[0]
            d={
                'classes':np.array(self.results.boxes.cls.cpu().numpy(),dtype=np.int8),
                'polygons':self.results.masks.xy
            }
            return d
        except:
            print('seg except')
            return None
    def get_results(self):
        return self.results
    def get_classes(self):
        return self.results.boxes.cls
    def get_bboxes(self):
        return self.results.boxes.xywh
    def get_normalized_bboxes(self):
        return self.results.boxes.xywhn
    def get_mask(self):
        return self.results.masks.data
    def get_polygons(self):
        return self.results.masks.xy
    def get_normalized_polygons(self):
        return self.results.masks.xyn
    def get_dictionary(self):
        d={
            'classes':np.array(self.results.boxes.cls,dtype=np.int8),
            'polygons':np.array(self.results.masks.xy,dtype=object)
        }
        return d

