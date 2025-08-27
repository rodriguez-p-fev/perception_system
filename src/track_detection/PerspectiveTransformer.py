import numpy as np
import cv2
'''
WIDTH = 4096.0
HEIGHT = 3040.0
GAUGE = 1435 #mM
SELECTED_LONGITUD = 1000 #mts
SOURCE  = np.array([
    [1820.0, 1500.0],
    [2380.0, 1500.0],
    [3025.0, 3040.0],
    [1125.0, 3040.0],
])
DEST1 = np.array([
    [WIDTH/2, HEIGHT/2],
    [WIDTH/2+GAUGE, HEIGHT/2],
    [WIDTH/2+GAUGE, HEIGHT/2 + SELECTED_LONGITUD-1],
    [WIDTH/2,HEIGHT/2 + SELECTED_LONGITUD-1],
])
DEST = np.array([
    [0, 0],
    [GAUGE-1, 0],
    [GAUGE-1, SELECTED_LONGITUD-1],
    [0, SELECTED_LONGITUD-1],
])
'''
class PerspectiveTransformer:
    #def __init__(self, source: np.ndarray, target: np.ndarray):
    def __init__(self, SOURCE, DEST):
        self.source  = SOURCE
        self.target = DEST
        self.M = cv2.getPerspectiveTransform(
            src = self.source.astype(np.float32),
            dst = self.target.astype(np.float32)
        )
        self.inverse = np.linalg.inv(self.M)
        #print(self.M)
        return None
    def transform(self, points: np.ndarray) -> np.ndarray:
        points = points.astype(np.float32)
        reshaped_points = points.reshape(-1, 1, 2)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.M)
        return transformed_points.reshape(-1, 2).astype(np.int32)
    def inverse_transform(self, points: np.ndarray) -> np.ndarray:
        points = points.astype(np.float32)
        reshaped_points = points.reshape(-1, 1, 2)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.inverse)
        return transformed_points.reshape(-1, 2).astype(np.int32)
