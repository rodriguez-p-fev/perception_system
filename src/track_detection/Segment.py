import os
import numpy as np
from track_detection import utils
from track_detection.Polygon import Polygon
from PIL import Image, ImageDraw, ImageFont

segment_classes = {
    0:"track",
    1:"fp_turnout",
    2:"tp_turnout",
    3:"common",
    4:'common-left',
    5:'common-right',
}
segment_states = {
    0:'unactive',
    1:'active',
    2:'unknown',
}
direction_dict = {
    -1:"left",
    0:"straight",
    1:"right",
    2:"unknown",
}
class Segment:
    def __init__(self, polygon: Polygon) -> object:
        self.segment_class = (polygon.get_class()+3)%5
        self.polygons = [polygon]
        self.direction = 2
        self.state = 0
        self.active_polygons_list = []
        self.bbox = polygon.get_bbox()
        self.update_base_points()
        return None
    def add_polygon(self, new_polygon: Polygon) -> None:
        self.polygons.append(new_polygon)
        return None
    def update_base_points(self):
        self.cx = (self.bbox[0]+self.bbox[2])/2
        self.cy = (self.bbox[1]+self.bbox[3])/2
        self.top_center_point  = [self.cx, self.bbox[1]]
        self.base_center_point = [self.cx, self.bbox[3]]
        self.search_distance = 10.1*(self.bbox[2] - self.bbox[0])*(self.bbox[3] - self.bbox[1])
        return None
    def update_bbox(self) -> None:
        if(len(self.polygons) == 1):
            self.bbox = self.polygons[0].get_bbox()
        else:
            x0, y0, x1, y1 = [], [], [], []
            for p in self.polygons:
                x0.append(p.get_bbox()[0])
                y0.append(p.get_bbox()[1])
                x1.append(p.get_bbox()[2])
                y1.append(p.get_bbox()[3])
            self.bbox = [np.array(x0).min(),np.array(y0).min(),np.array(x1).max(),np.array(y1).max()]
        self.update_base_points()
        self.sort_polygons()
        return None
    def get_active_bbox(self):
        if(self.direction == -1):
            active_bbox = [min(self.polygons[0].get_bbox()[0],self.polygons[1].get_bbox()[0]),
                            min(self.polygons[0].get_bbox()[1],self.polygons[1].get_bbox()[1]),
                            max(self.polygons[0].get_bbox()[2],self.polygons[1].get_bbox()[2]),
                            max(self.polygons[0].get_bbox()[3],self.polygons[1].get_bbox()[3])]
        elif(self.direction == 1):
            active_bbox = [min(self.polygons[0].get_bbox()[0],self.polygons[2].get_bbox()[0]),
                            min(self.polygons[0].get_bbox()[1],self.polygons[2].get_bbox()[1]),
                            max(self.polygons[0].get_bbox()[2],self.polygons[2].get_bbox()[2]),
                            max(self.polygons[0].get_bbox()[3],self.polygons[2].get_bbox()[3])]
        else:
            active_bbox = self.polygons[0].get_bbox()
        return active_bbox
    def sort_polygons(self):
        common_idx = 0
        legs_idxs = []
        if(len(self.polygons) == 3):
            for enum, p in enumerate(self.polygons):
                if(p.get_class() == 0):
                    common_idx = enum
                else:
                    legs_idxs.append(enum)
            if(self.polygons[legs_idxs[0]].get_center_point()[0] < self.polygons[legs_idxs[1]].get_center_point()[0]):
                left_leg_idx = legs_idxs[0]
                right_leg_idx = legs_idxs[1]
            else:
                left_leg_idx = legs_idxs[1]
                right_leg_idx = legs_idxs[0]
            self.polygons = [self.polygons[common_idx],self.polygons[left_leg_idx],self.polygons[right_leg_idx]]
        elif(len(self.polygons) == 2):
            for enum, p in enumerate(self.polygons):
                if(p.get_class() == 0):
                    common_idx = enum
                else:
                    legs_idxs.append(enum)
            if(self.polygons[legs_idxs[0]].get_center_point()[0] < self.polygons[common_idx].get_center_point()[0]):
                left_leg_idx = legs_idxs[0]
                right_leg_idx = common_idx
            else:
                left_leg_idx = common_idx
                right_leg_idx = legs_idxs[0]
            self.polygons = [self.polygons[common_idx],self.polygons[left_leg_idx],self.polygons[right_leg_idx]]
        return None
    def set_activation(self, state):
        if(state == 1):
            self.activate()
        else:
            self.set_polygons_state(state)
        return None
    def set_direction(self, direction: int):
        self.direction = direction
        return None
    def set_class(self, segment_class: int):
        self.segment_class = segment_class
        return None
    def set_state(self, new_state: int) -> None:
        self.state = new_state
        for p in self.polygons:
            p.set_state(self.state)
        return None
    def get_state(self) -> int:
        return self.state
    def activate(self):
        if(segment_classes[self.segment_class] in ['track','turnout']):
            self.polygons[0].set_state(1)
            self.active_polygons_list.append(self.polygons[0])
        elif(self.segment_class in [1,2]):
            if(self.direction == -1):
                if(len(self.polygons) == 3):
                    self.polygons[0].set_state(1)
                    self.polygons[1].set_state(1)
                    self.polygons[2].set_state(0)
                    self.active_polygons_list.extend([self.polygons[0],self.polygons[1]])
                else:
                    self.polygons[0].set_state(1)
            elif(self.direction == 1):
                if(len(self.polygons) == 3):
                    self.polygons[0].set_state(1)
                    self.polygons[2].set_state(1)
                    self.polygons[1].set_state(0)
                    self.active_polygons_list.extend([self.polygons[0],self.polygons[2]])
                else:
                    self.polygons[0].set_state(1)
                    self.active_polygons_list.append(self.polygons[0])
        return None
    def set_polygons_state(self, state):
        for p in self.polygons:
            p.set_state(state)
        return None
    def update_keypoints(self, objects):
        intersections = []
        idxs = []
        for enum, obj in enumerate(objects):
            if(self.segment_class == obj.get_class() or (self.segment_class > 0 and obj.get_class() > 0)):
                intersections.append(utils.bboxes_IoU(self.get_bbox(),obj.get_bbox()))
                idxs.append(enum)
        intersections = np.array(intersections)
        if(len(intersections) > 0 and intersections.max() > 30):
            obj_idx = idxs[np.where(intersections == intersections.max())[0][0]]
            self.segment = objects[obj_idx]
            self.segment_class = objects[obj_idx].get_class()
            self.pose = objects[obj_idx].get_pose()
            self.object_bbox = objects[obj_idx].get_bbox()
            self.direction = objects[obj_idx].get_direction()
        else:
            self.object_bbox = []
            self.pose = []
        return None
    def get_class(self):
        return self.segment_class
    def get_bbox(self):
        return self.bbox
    def get_polygons(self):
        return self.polygons
    def get_direction(self):
        return self.direction
    def get_center_point(self):
        return [self.cx,self.cy]
    def get_conection_points(self):
        if(segment_classes[self.segment_class] not in ['fp_turnout','tp_turnout']):
            return [
                self.polygons[0].get_weighted_bottom_center_point(),
                self.polygons[0].get_weighted_top_center_point()
            ]
        elif(segment_classes[self.segment_class] == 'fp_turnout'):
            if(len(self.polygons) == 3):
                return [
                    self.polygons[0].get_weighted_bottom_center_point(),
                    self.polygons[1].get_weighted_top_center_point(),
                    self.polygons[2].get_weighted_top_center_point()
                ]
            elif(len(self.polygons) == 2):
                return [
                    self.polygons[0].get_weighted_bottom_center_point(),
                    self.polygons[1].get_weighted_top_center_point(),
                    self.polygons[0].get_weighted_top_center_point()
                ]
            else:
                return [
                    self.polygons[0].get_weighted_bottom_center_point(),
                    self.polygons[0].get_weighted_top_center_point(),
                    self.polygons[0].get_weighted_top_center_point()
                ]
        elif(segment_classes[self.segment_class] == 'tp_turnout'):
            if(len(self.polygons) == 3):
                return [
                    self.polygons[1].get_weighted_bottom_center_point(),
                    self.polygons[2].get_weighted_bottom_center_point(),
                    self.polygons[0].get_weighted_top_center_point()
                ]
            elif(len(self.polygons) == 2):
                return [
                    self.polygons[1].get_weighted_bottom_center_point(),
                    self.polygons[0].get_weighted_bottom_center_point(),
                    self.polygons[0].get_weighted_top_center_point()
                ]
            else:
                return [
                    self.polygons[0].get_weighted_bottom_center_point(),
                    self.polygons[0].get_weighted_bottom_center_point(),
                    self.polygons[0].get_weighted_top_center_point()
                ]
    def get_bottom_search_distance(self):
        return max(450,1.25*self.polygons[0].get_bottom_width())