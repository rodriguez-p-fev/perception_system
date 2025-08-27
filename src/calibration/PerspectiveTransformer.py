import numpy as np
import cv2

class PerspectiveTransformer:
    def __init__(self, SOURCE, DEST):
        self.source  = SOURCE
        self.target = DEST
        self.M = cv2.getPerspectiveTransform(
            src = self.source.astype(np.float32),
            dst = self.target.astype(np.float32)
        )
        self.inverse = np.linalg.inv(self.M)
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
    def get_transformation_matrix(self):
        return self.M
