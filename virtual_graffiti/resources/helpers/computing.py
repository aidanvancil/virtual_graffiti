import cv2
import numpy as np

def count_filled_pixels(canvas, background_image):
    filled_pixels = np.sum(np.any(canvas != [0, 0, 0], axis=2))
    total_pixels = background_image.shape[0] * background_image.shape[1]
    return filled_pixels, total_pixels

def smooth_drawing(last_point, current_point, canvas, color=(0, 0, 255), thickness=2, distance_threshold=160):
    if last_point is not None and np.linalg.norm(np.array(last_point) - np.array(current_point)) < distance_threshold: # and is_registered(color):
        cv2.line(canvas, last_point, current_point, color, thickness, lineType=cv2.LINE_AA)
    return current_point

def skew_point(point, skew_matrix):
    homogeneous_point = np.array([point[0], point[1], 1])
    skewed_point = np.dot(skew_matrix, homogeneous_point)
    return (int(skewed_point[0] / skewed_point[2]), int(skewed_point[1] / skewed_point[2]))