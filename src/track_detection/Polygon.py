import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from track_detection import utils
from track_detection import PerspectiveTransformer

directory = os.path.dirname(os.path.abspath(__file__))
font_path = os.path.join(directory, 'resources/swansea-font/Swansea-q3pd.ttf')
font = ImageFont.truetype(font_path, 80)

polygon_classes = {
    2:'track',
    1:'leg',
    0:'common'
}
polygon_states = {
    'unactive': 0,
    'active': 1,
    'unknown':2
}

class Polygon:
    def __init__(self, polygon_class: int, polygon: np.ndarray, perspective_transformer: PerspectiveTransformer) -> object:
        self.polygon_class = polygon_class
        self.polygon = polygon
        self.xs = self.polygon[:,0]
        self.ys = self.polygon[:,1]
        self.state = 2
        self.initialize_polygon()
        self.perspective_transformer = perspective_transformer
        return None
    def initialize_polygon(self) -> None:
        cut_percentage = 0.15
        self.left = self.xs.min()
        self.top = self.ys.min()
        self.right = self.xs.max()
        self.bottom = self.ys.max()
        self.cx = (self.left+self.right)/2
        self.cy = (self.top+self.bottom)/2
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
    def get_perspective_polygon(self):
        point_set =  np.array(self.polygon, dtype=np.int32)
        point_set =  self.perspective_transformer.transform(point_set, )
        return point_set
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
    def draw_polygon(self, img: Image, color_rgb=[0,255,0]):
        img_array = np.array(img)
        output = np.copy(img_array)
        alpha = 0.4
        point_set =  np.array(self.polygon, dtype=np.int32)
        cv2.fillPoly(output, pts=[point_set], color=color_rgb)
        image_new = cv2.addWeighted(output, alpha, img_array, 1 - alpha, 0)
        image_new = Image.fromarray(image_new)
        #self.draw_pointset(image_new, [[self.wx_bottom,self.bottom],[self.wx_top,self.top]])
        return image_new
    def draw_state_polygon(self, img: Image):
        if(self.state == 0):
            color_rgb=[0,0,255]
        elif(self.state == 1):
            color_rgb=[0,255,0]
        else:
            color_rgb=[255,255,0]
        img_array = np.array(img)
        output = np.copy(img_array)
        alpha = 0.4
        point_set =  np.array(self.polygon, dtype=np.int32)
        #point_set =  self.perspective_transformer.transform_points(point_set)
        #point_set =  utils.rescale_pointset(point_set)
        #print(point_set)
        cv2.fillPoly(output, pts=[point_set], color=color_rgb)
        image_new = cv2.addWeighted(output, alpha, img_array, 1 - alpha, 0)
        image_new = Image.fromarray(image_new)
        return image_new
    '''
    def initialize_polygon(self) -> None:
        cut_percentage = 0.15
        self.left = self.xs.min()
        self.top = self.ys.min()
        self.right = self.xs.max()
        self.bottom = self.ys.max()
        #self.cx = (self.left+self.right)/2
        #self.cy = (self.top+self.bottom)/2
        #self.wx = self.xs.mean()
        #self.wy = self.ys.mean()
        self.width = self.right - self.left
        self.height = self.bottom - self.top
        #cut_x_left = self.left+self.width*cut_percentage
        #cut_x_right = self.right-self.width*cut_percentage
        cut_y_top = self.top+self.height*cut_percentage
        cut_y_bottom = self.bottom-self.height*cut_percentage
        self.wx_top = self.polygon[np.where(self.ys <= cut_y_top)][:,0].mean()
        self.wx_bottom = self.polygon[np.where(self.ys > cut_y_bottom)][:,0].mean()
        #self.wy_left = self.polygon[np.where(self.xs <= cut_x_left)][:,1].mean()
        #self.wy_right = self.polygon[np.where(self.xs > cut_x_right)][:,1].mean()
        #self.xdiff_top = self.polygon[np.where(self.ys <= cut_y_top)][:,0].max() - self.polygon[np.where(self.ys <= cut_y_top)][:,0].min()
        #self.xdiff_bottom = self.polygon[np.where(self.ys > cut_y_bottom)][:,0].max() - self.polygon[np.where(self.ys > cut_y_bottom)][:,0].min()
        return None
    def get_center_point(self):
        return [self.cx,self.cy]
    def get_weighted_center_point(self):
        return [self.wx,self.wy]
    def draw_state_polygon(self, img: Image):
        if(self.state == 0):
            color_rgb=[0,0,255]
        elif(self.state == 1):
            color_rgb=[0,255,0]
        else:
            color_rgb=[255,255,0]
        img_array = np.array(img)
        output = np.copy(img_array)
        alpha = 0.4
        point_set =  np.array(self.polygon, dtype=np.int32)
        #point_set =  self.perspective_transformer.transform_points(point_set)
        #point_set =  utils.rescale_pointset(point_set)
        #print(point_set)
        cv2.fillPoly(output, pts=[point_set], color=color_rgb)
        image_new = cv2.addWeighted(output, alpha, img_array, 1 - alpha, 0)
        image_new = Image.fromarray(image_new)
        return image_new
    def draw_polygon(self, img: Image, color_rgb=[0,255,0]):
        img_array = np.array(img)
        output = np.copy(img_array)
        alpha = 0.4
        point_set =  np.array(self.polygon, dtype=np.int32)
        cv2.fillPoly(output, pts=[point_set], color=color_rgb)
        image_new = cv2.addWeighted(output, alpha, img_array, 1 - alpha, 0)
        image_new = Image.fromarray(image_new)
        #self.draw_pointset(image_new, [[self.wx_bottom,self.bottom],[self.wx_top,self.top]])
        return image_new
    def draw_plain_bbox(self, img: Image, bbox, color_rgb=[0,255,0]):
        drw = ImageDraw.Draw(img, 'RGB')         
        shape = [(bbox[0],bbox[1]),(bbox[2],bbox[3])]
        drw.rectangle(shape, outline=(color_rgb[0],color_rgb[1],color_rgb[2]), width=10)
        return img
    def draw_bbox(self, img: Image, bbox, class_name, color_rgb=[0,255,0]):
        drw = ImageDraw.Draw(img, 'RGB')         
        shape = [(bbox[0],bbox[1]),(bbox[2],bbox[3])]
        text_bbox = drw.textbbox((bbox[0],bbox[1]-80), class_name, font=font)
        drw.rectangle(text_bbox, fill=(color_rgb[0],color_rgb[1],color_rgb[2]))
        drw.text((bbox[0],bbox[1]-80), class_name, fill=(0,0,0), font=font)
        drw.rectangle(shape, outline=(color_rgb[0],color_rgb[1],color_rgb[2]), width=5)
        return img
    def draw_pointset(self, img: Image, pointset, color_rgb=[0,255,0], width = 15):
        drw = ImageDraw.Draw(img, 'RGB') 
        for p in pointset:
            drw.ellipse((p[0]-width,p[1]-width, p[0]+width, p[1]+width), fill=(color_rgb[0],color_rgb[1],color_rgb[2]), outline =(color_rgb[0],color_rgb[1],color_rgb[2]))
        return img
    '''