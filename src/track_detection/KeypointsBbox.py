import numpy as np
import os
from track_detection import utils
from PIL import Image, ImageDraw, ImageFont


object_classes = {
    0:"track",
    1:"fp_turnout",
    2:"tp_turnout"
}
direction_dict = {
    -1:"left",
    0:"straight",
    1:"right",
    2:"unknown"
}
class KeypointsBbox:
    def __init__(self, bbox_class, bbox, keypoints):
        self.bbox_class = bbox_class
        self.bbox = bbox
        self.keypoints = keypoints
        self.direction = 2
        self.update_pose()
        return None
    def update_pose(self):
        if(self.bbox_class == 0):
            left_bottom = self.keypoints[0]
            right_bottom = self.keypoints[1]
            right_top = self.keypoints[2]
            left_top = self.keypoints[3]
            self.pose = [left_bottom, right_bottom, right_top, left_top]
        else:
            self.sort_pose(self.keypoints)
        return None
    def sort_pose(self, pointset):
        if(pointset[0][0] < pointset[2][0]):
            left1 = pointset[0]
            left2 = pointset[5]
            right1 = pointset[2]
            right2 = pointset[3]
            self.direction = -1
        else:
            left1 = pointset[2]
            left2 = pointset[3]
            right1 = pointset[0]
            right2 = pointset[5]
            self.direction = 1
        if(left1[1] > left2[1]):
            left_bottom = left1
            left_top = left2
        else:
            left_bottom = left2
            left_top = left1
            self.bbox_class = 2
        if(right1[1] > right2[1]):
            right_bottom = right1
            right_top = right2
        else:
            right_bottom = right2
            right_top = right1
            self.bbox_class = 2
        self.pose = [left_bottom,right_bottom,right_top,left_top,pointset[4]]
        return None
    def get_class(self):
        return self.bbox_class
    def get_bbox(self):
        return self.bbox
    def get_keypoints(self):
        return self.keypoints
    def get_direction(self):
        return self.direction
    def get_pose(self):
        return self.pose