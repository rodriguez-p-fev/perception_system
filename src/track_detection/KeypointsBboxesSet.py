import numpy as np
import random
from PIL import Image, ImageDraw
from track_detection.KeypointsBbox import KeypointsBbox

class KeypointsBboxesSet:
    def __init__(self, keypoints_model_output: dict) -> object:
        self.keypointsbboxes_dict = keypoints_model_output
        self.bboxes_graph = []
        self.initialize_bboxes_list()
        return None
    def initialize_bboxes_list(self) -> None:
        self.bboxes_classes = []
        self.keypoints_bboxes = []
        for enum, c in enumerate(self.keypointsbboxes_dict['classes']):
            bbox = self.keypointsbboxes_dict['bboxes'][enum]
            keypoints = self.keypointsbboxes_dict['keypoints'][enum]
            self.bboxes_classes.append(c)
            new_object = KeypointsBbox(c, bbox, keypoints)
            new_object.update_pose()
            self.keypoints_bboxes.append(new_object)
        return None 
    def get_keypoints_bboxes(self) -> list:
        return self.keypoints_bboxes
    def initialize_graph(self) -> None:
        self.find_start_node()
        for enum, obj in  enumerate(self.keypoints_bboxes):
            self.objects_graph.append(obj.get_next_nodes(self.keypoints_bboxes))
        return None
    def find_start_node(self) -> None:
        y_coords = []
        idxs = []
        for enum, object in enumerate(self.keypoints_bboxes):
            y_coords.append(object.get_bbox()[3])
            idxs.append(enum)
        sorted_idxs = np.argsort(y_coords)
        self.start_node = idxs[sorted_idxs[-1]]
        return None
    def update_active_path(self) -> None:
        actual = self.start_node
        next_nodes = self.objects_graph[actual]
        while actual != -1:
            if(self.keypoints_bboxes[actual].get_class() in [0,2]):
                self.keypoints_bboxes[actual].get_segment().get_polygons()[0].set_state(1)
                actual = next_nodes[0]
                next_nodes = self.objects_graph[actual]
            if(self.objects[actual].get_class() == 1):
                if(self.keypoints_bboxes[actual].get_direction() == -1):
                    self.keypoints_bboxes[actual].get_segment().get_polygons()[0].set_state(1)
                    self.keypoints_bboxes[actual].get_segment().get_polygons()[1].set_state(1)
                    self.keypoints_bboxes[actual].get_segment().get_polygons()[2].set_state(0)
                    self.keypoints_bboxes[next_nodes[1]].get_segment().get_polygons()[0].set_state(0)
                    actual = next_nodes[0]
                    next_nodes = self.objects_graph[actual]
                else:
                    self.keypoints_bboxes[actual].get_segment().get_polygons()[0].set_state(1)
                    self.keypoints_bboxes[actual].get_segment().get_polygons()[1].set_state(0)
                    self.keypoints_bboxes[actual].get_segment().get_polygons()[2].set_state(1)
                    self.keypoints_bboxes[next_nodes[0]].get_segment().get_polygons()[0].set_state(0)
                    actual = next_nodes[1]
                    next_nodes = self.objects_graph[actual]
        return None


