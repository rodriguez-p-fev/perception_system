import os
import numpy as np

segmentation_model = {
    0:'common',
    1:'leg',
    2:'track'
}
class SegmentationInference:
    def __init__(self, model_output_files_path: str, output_files: list) -> object:
        self.files_path=model_output_files_path
        self.output_files = output_files
        return None
    def get_inference(self, idx: int, img_shape: np.ndarray) -> dict:
        try:
            inference_file_path = os.path.join(self.files_path, self.output_files[idx])
            polygons_str = self.get_polygons_from_path(inference_file_path)
            return self.parse_polygon_str_lines(polygons_str, img_shape)
        except:
            return None
    def get_polygons_from_path(self, file_path: str) -> list:
        f = open(file_path)
        polygons = []
        for readline in f:
            array = readline.split(' ')
            polygons.append(array)
        f.close()
        return polygons
    def parse_polygon_str_lines(self, polygons_str: list, img_shape: np.ndarray) -> dict:
        polygons_classes = []
        polygons = []
        for polygon_str_line in polygons_str:
            polygon_class = int(polygon_str_line[0])
            polygon = np.array(polygon_str_line[1:],dtype=np.float32)
            polygon = img_shape*np.reshape(polygon,(int(len(polygon)/2),2),order='A')

            polygons_classes.append(polygon_class)
            polygons.append(polygon)
        d={
            'classes':np.array(polygons_classes,dtype=np.int8),
            'polygons':np.array(polygons,dtype=object)
        }
        return d
