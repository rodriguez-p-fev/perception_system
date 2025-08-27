import os
import numpy as np
import cv2
from PIL import Image
from track_detection import utils

polygon_classes = {
    2:'track',
    1:'leg',
    0:'common'
}
polygon_states = {
    0:'unactive',
    1:'active',
    2:'unknown'
}

class Polygon:
    def __init__(self, polygon_class: int, polygon: np.ndarray) -> object:
        self.polygon_class = polygon_class
        self.polygon = polygon
        self.xs = self.polygon[:,0]
        self.ys = self.polygon[:,1]
        self.state = 2
        self.initialize_polygon()
        return None
    def initialize_polygon(self) -> None:
        cut_percentage = 0.15
        self.left = self.xs.min()
        self.top = self.ys.min()
        self.right = self.xs.max()
        self.bottom = self.ys.max()
        self.cx = (self.left+self.right)/2
        self.cy = (self.top+self.bottom)/2
        self.wx = self.polygon[:,0].mean()
        self.wx = self.polygon[:,0].mean()
        self.width = self.right - self.left
        self.height = self.bottom - self.top
        cut_y_top = self.top+self.height*cut_percentage
        cut_y_bottom = self.bottom-self.height*cut_percentage
        self.wx_top = self.polygon[np.where(self.ys <= cut_y_top)][:,0].mean()
        self.wx_bottom = self.polygon[np.where(self.ys > cut_y_bottom)][:,0].mean()
        self.xdiff_top = self.polygon[np.where(self.ys <= cut_y_top)][:,0].max() - self.polygon[np.where(self.ys <= cut_y_top)][:,0].min()
        self.xdiff_bottom = self.polygon[np.where(self.ys > cut_y_bottom)][:,0].max() - self.polygon[np.where(self.ys > cut_y_bottom)][:,0].min()
        return None
    def set_state(self, new_state: int) -> None:
        self.state = new_state
        return None
    def get_class(self) -> int:
        return self.polygon_class
    def get_polygon(self) -> np.ndarray:
        return self.polygon
    def get_state(self) -> int:
        return self.state
    def get_bbox(self) -> list:
        return [self.left,self.top,self.right,self.bottom]
    def get_weighted_bottom_center_point(self) -> list:
        return [self.wx_bottom,self.bottom]
    def get_weighted_top_center_point(self) -> list:
        return [self.wx_top,self.top]
    def get_center_point(self):
        return [self.cx,self.cy]
    def get_bottom_width(self):
        return self.xdiff_bottom
    def get_width(self):
        return self.width
