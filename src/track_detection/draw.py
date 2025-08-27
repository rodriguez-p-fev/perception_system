import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
directory = os.path.dirname(os.path.abspath(__file__))
font_path = os.path.join(directory, 'resources/swansea-font/Swansea-q3pd.ttf')
font = ImageFont.truetype(font_path, 80)

def draw_polygon(img, polygon, color_rgb=[0,255,0]):
    output = np.copy(img)
    alpha = 0.4
    point_set =  np.array(polygon, dtype=np.int32)
    cv2.fillPoly(output, pts=[point_set], color=color_rgb)
    output_image = cv2.addWeighted(output, alpha, img, 1 - alpha, 0)
    output_image = Image.fromarray(output_image)
    return output_image
def draw_segments(img_array, polygons, color_rgb):
    output_image = np.copy(img_array)
    for enum, p in enumerate(polygons):
        output_image = draw_polygon(output_image, p, color_rgb)
    output_image = Image.fromarray(output_image)
    return output_image
def draw_foul_volume_lines(img, lines, line_color, fig='lines'):
    drw = ImageDraw.Draw(img, 'RGB')
    for l in lines:
        for i in range(1,len(l)):
            if(fig=='lines'):
                drw.line([tuple(l[i-1]),tuple(l[i])], fill=line_color, width=5)
            if(fig=='points'):
                drw.circle(xy=tuple(l[i]), radius=5, fill = line_color, outline =line_color, width=1)
    return img
def draw_bbox(img, bbox, color, text_bbox=''):
    class_name = 'class_dictionary[str(bbox_class)]'
    shape = [(bbox[0],bbox[1]),(bbox[2],bbox[3])]
    draw = ImageDraw.Draw(img, 'RGB') 
    text_bbox = draw.textbbox((bbox[0][0],bbox[0][1]-80), class_name, font=font)
    draw.rectangle(text_bbox, fill=color)
    draw.text((bbox[0][0],bbox[0][1]-80), class_name, fill=(0,0,0), font=font)
    draw.rectangle(shape, outline=color, width=5)
    return img
def draw_pointset(img, pointset, color=[255,0,0]):
    drw = ImageDraw.Draw(img, 'RGB') 
    for p in pointset:
        drw.ellipse((p[0],p[1], p[0]+10, p[1]+10), fill = 'red', outline ='red')
    return img
def draw_maks_from_polygons(img, shapes):
    blank = np.zeros(shape=(img.shape[0],img.shape[1]),dtype=np.float32)
    for shape in shapes:
        points = np.array(shape,np.int32)
        cv2.fillPoly(blank,[points],255)
    mask_array = np.array(blank)
    mask_array = np.uint8(mask_array)
    output_image = Image.fromarray(mask_array)
    return output_image

