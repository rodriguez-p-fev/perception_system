import numpy as np
import random
from track_detection import Segment
from track_detection import utils
from track_detection.Polygon import Polygon
from calibration import PerspectiveTransformer
from PIL import Image

class SegmentsSet:
    def __init__(self, segmentation_model_output: dict, perspective_transformer: PerspectiveTransformer) -> object:
        self.segmentation_dict = segmentation_model_output
        self.perspective_transformer = perspective_transformer
        self.segments = []
        self.nodes_graph = []
        self.active = []
        self.initialize_polygon_list()
        return None
    
    #Initialize polygons
    def initialize_polygon_list(self) -> None:
        self.polygons_classes = []
        self.polygons = []
        for enum, c in enumerate(self.segmentation_dict['classes']):
            self.polygons_classes.append(c)
            self.polygons.append(Polygon(c, self.segmentation_dict['polygons'][enum]))
        self.polygons_classes = np.array(self.polygons_classes)
        self.polygons = np.array(self.polygons,dtype='object')
        self.common_idxs = np.where(self.polygons_classes == 0)[0]
        self.leg_idxs = np.where(self.polygons_classes == 1)[0]
        self.track_idxs = np.where(self.polygons_classes == 2)[0]
        return None
    
    #Initialize segments
    def initialize_segments(self, keypointsbboxes_list) -> None:
        segments_base_center_points = []
        self.unknown_nodes = []
        tracks = self.polygons[self.track_idxs]
        commons = self.polygons[self.common_idxs]
        legs = self.polygons[self.leg_idxs]
        for common in commons:
            new_segment = self.select_legs(common, legs)
            #new_segment = Segment.Segment(common)
            new_segment.update_bbox()
            new_segment.update_keypoints(keypointsbboxes_list)
            self.segments.append(new_segment)
            self.nodes_graph.append([])
            segments_base_center_points.append(new_segment.get_center_point())
        for track in tracks:
            new_segment = Segment.Segment(track)
            self.segments.append(new_segment)
            self.nodes_graph.append([])
            segments_base_center_points.append(new_segment.get_center_point())
        segments_base_center_points = np.array(segments_base_center_points)
        self.sorted_segments_idxs = np.flip(np.argsort(segments_base_center_points[:,1]))
        return None
    def select_legs(self, common:Polygon, legs:list):
        new_segment = Segment.Segment(common)
        intersections = []
        distances = []
        for leg in legs:
            intersections.append(utils.bboxes_IoU(common.get_bbox(),leg.get_bbox()))
            distances.append(utils.distance(common.get_center_point(),leg.get_center_point()))
        
        intersections = np.array(intersections,dtype=np.float16)
        intersections_idx = np.where(intersections>0)[0]
        intersections = intersections[intersections_idx]

        distances = np.array(distances)
        distances_idx = np.where(distances<1.25*common.get_width())[0]
        distances = distances[distances_idx]
        
        if(len(intersections_idx) > 0):
            sorted_idxs = np.flip(np.argsort(intersections))
            if(len(intersections_idx) >= 2):
                new_segment.add_polygon(legs[intersections_idx[sorted_idxs[0]]])
                new_segment.add_polygon(legs[intersections_idx[sorted_idxs[1]]])
                if(common.get_center_point()[1] >= (legs[sorted_idxs[0]].get_center_point()[1]+legs[sorted_idxs[1]].get_center_point()[1])/2):
                    new_segment.set_class(1)
                elif(common.get_center_point()[1] < (legs[sorted_idxs[0]].get_center_point()[1]+legs[sorted_idxs[1]].get_center_point()[1])/2): 
                    new_segment.set_class(2)
            elif(len(intersections) == 1):
                new_segment.add_polygon(legs[sorted_idxs[0]])
                if(common.get_center_point()[0] >= legs[sorted_idxs[0]].get_center_point()[0]):
                    new_segment.set_class(4)
                elif(common.get_center_point()[0] < legs[sorted_idxs[0]].get_center_point()[0]):
                    new_segment.set_class(5)
        return new_segment
    
    #Join keypoints model output
    def set_nodes_graph(self, start_point: np.ndarray):
        self.start_node = self.get_closer_node(start_point, weight=2.6)
        for i in range(len(self.sorted_segments_idxs)-1):
            idx = self.sorted_segments_idxs[i]
            if(Segment.segment_classes[self.segments[idx].get_class()] not in ['fp_turnout','tp_turnout']):
                top_point = self.segments[idx].get_conection_points()[1]
                selected_node = self.get_next_node(top_point, i + 1)
                self.nodes_graph[idx].append(selected_node)
            elif(Segment.segment_classes[self.segments[idx].get_class()] == 'tp_turnout'):
                top_point = self.segments[idx].get_conection_points()[2]
                selected_node = self.get_next_node(top_point, i + 1)
                self.nodes_graph[idx].append(selected_node)
            elif(Segment.segment_classes[self.segments[idx].get_class()] == 'fp_turnout'):
                top_point_1 = self.segments[idx].get_conection_points()[1]
                top_point_2 = self.segments[idx].get_conection_points()[2]
                selected_node = self.get_next_node(top_point_1, i + 1)
                self.nodes_graph[idx].append(selected_node)
                selected_node = self.get_next_node(top_point_2, i + 1)
                self.nodes_graph[idx].append(selected_node)
        if(Segment.segment_classes[self.segments[self.sorted_segments_idxs[-1]].get_class()] == 'fp_turnout'):
            self.nodes_graph[self.sorted_segments_idxs[-1]].append(-1)
            self.nodes_graph[self.sorted_segments_idxs[-1]].append(-1)
        else:
            self.nodes_graph[self.sorted_segments_idxs[-1]].append(-1)
        print(self.nodes_graph)
        return None
    
    #Set active path
    def set_active_path(self, start):
        node = start
        self.active_nodes = []
        self.active_polygons = []
        while node != -1 and self.get_segments()[node].get_state() != 2:
            if(len(self.nodes_graph[node]) == 2 and self.nodes_graph[node][0] != self.nodes_graph[node][1]):
                self.get_segments()[node].set_activation(1)
                self.active_polygons.extend(self.get_segments()[node].get_polygons()[0].get_polygon())
                self.active_nodes.append(node)
                if(self.get_segments()[node].get_direction() == -1):
                    self.active_polygons.extend(self.get_segments()[node].get_polygons()[1].get_polygon())
                    node = self.nodes_graph[node][0]
                else:
                    self.active_polygons.extend(self.get_segments()[node].get_polygons()[2].get_polygon())
                    node = self.nodes_graph[node][1]
            else:
                self.get_segments()[node].set_activation(1)
                self.active_nodes.append(node)
                self.active_polygons.extend(self.get_segments()[node].get_polygons()[0].get_polygon())
                node = self.nodes_graph[node][0]
        return self.active_nodes
    def get_active_polygons(self, active_nodes):
        self.active_polygons = []
        for node in active_nodes:
            if(self.segments[node].get_class() in [0,3]):
                self.active_polygons.append(self.segments[node].get_polygons()[0].get_polygon())
            elif(self.segments[node].get_class() in [1,2]):
                self.active_polygons.append(self.segments[node].get_polygons()[0].get_polygon())
                if(self.get_segments()[node].get_direction() == -1):
                    self.active_polygons.append(self.segments[node].get_polygons()[1].get_polygon())
                else:
                    self.active_polygons.append(self.segments[node].get_polygons()[2].get_polygon())
        return self.active_polygons
    def set_path(self, start, state):
        node = start
        while node != -1:
            if(len(self.nodes_graph[node]) == 2 and self.nodes_graph[node][0] != self.nodes_graph[node][1]):
                self.get_segments()[node].set_activation(state)
                if(self.get_segments()[node].get_direction() == -1):
                    self.set_path(self.nodes_graph[node][1], 0)
                    node = self.nodes_graph[node][0]
                else:
                    self.set_path(self.nodes_graph[node][0], 0)
                    node = self.nodes_graph[node][1]
            else:
                self.get_segments()[node].set_activation(state)
                node = self.nodes_graph[node][0]
        return None
    
    #Start node functions
    def get_next_node(self, segment, start_idx):
        top_point = segment
        distances = []
        for enum, comp_idx in enumerate(self.sorted_segments_idxs[start_idx:]):
            if(Segment.segment_classes[self.segments[comp_idx].get_class()] != 'tp_turnout'):
                bottom_point = self.segments[comp_idx].get_conection_points()[0]
                d = utils.vertical_weighted_distance(top_point, bottom_point)
                distances.append(d)
            else:
                bottom_point_1 = self.segments[comp_idx].get_conection_points()[0]
                bottom_point_2 = self.segments[comp_idx].get_conection_points()[1]
                d_1 = utils.vertical_weighted_distance(top_point, bottom_point_1)
                d_2 = utils.vertical_weighted_distance(top_point, bottom_point_2)
                distances.append(min(d_1,d_2))
        if(len(distances)>0):
            d_min = min(distances)
            selected_idx = np.where(np.array(distances)==d_min)[0][0] + start_idx
            selected_node = self.sorted_segments_idxs[selected_idx]
            if(d_min < self.segments[selected_node].get_bottom_search_distance()):
                return selected_node
            else:
                return -1
        else:
            return -1
    def get_closer_node(self, top_point, start_idx = 0, weight = 1.8):
        distances = []
        for enum, comp_idx in enumerate(self.sorted_segments_idxs[start_idx:]):
            if(Segment.segment_classes[self.segments[comp_idx].get_class()] != 'tp_turnout'):
                bottom_point = self.segments[comp_idx].get_conection_points()[0]
                d = utils.vertical_weighted_distance(top_point, bottom_point, weight)
                distances.append(d)
            else:
                bottom_point_1 = self.segments[comp_idx].get_conection_points()[0]
                bottom_point_2 = self.segments[comp_idx].get_conection_points()[1]
                d_1 = utils.vertical_weighted_distance(top_point, bottom_point_1, weight)
                d_2 = utils.vertical_weighted_distance(top_point, bottom_point_2, weight)
                distances.append(min(d_1,d_2))
        d_min = min(distances)
        selected_idx = np.where(np.array(distances)==d_min)[0][0] + start_idx
        selected_node = self.sorted_segments_idxs[selected_idx]
        return selected_node
    
    #Getters
    def get_segments(self):
        return self.segments
    def get_segments_classes(self):
        return self.segments_classes
    def get_start_node(self):
        return self.start_node
    def get_active_nodes(self):
        return self.active_nodes
    def get_start_segment(self):
        return self.segments[self.start_node]
    
    #Draw functions to delete
    def draw_bboxes(self, img: Image) -> Image:
        for s in self.get_segments():
            random_color = [random.randint(80,255),random.randint(80,255),random.randint(80,255)]
            img = s.draw_polygons(img, random_color)
            img = s.draw_bbox(img, random_color)
        return img
    def draw_active_segments(self, img: Image) -> Image:
        for enum, s in enumerate(self.get_segments()):
            img = s.draw_activations(img)
        return img
    