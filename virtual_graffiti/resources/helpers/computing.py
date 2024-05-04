import cv2
import numpy as np

def count_filled_pixels(canvas, background_image):
    """
    Counts the number of filled pixels on the canvas.

    Parameters:
        canvas (numpy.ndarray): A numpy array representing the canvas image.
        background_image (numpy.ndarray): A numpy array representing the background image.

    Returns:
        tuple: A tuple containing the number of filled pixels on the canvas and the total number of pixels in the background image.
    """
    filled_pixels = np.sum(np.any(canvas != [0, 0, 0], axis=2))
    total_pixels = background_image.shape[0] * background_image.shape[1]
    return filled_pixels, total_pixels

def smooth_drawing(last_point, current_point, canvas, color=(0, 0, 255), thickness=2, distance_threshold=160):
    """
    Smoothly draws a line between two points on the canvas if the distance between them is within a threshold.

    Parameters:
        last_point (tuple): Coordinates of the last point where the line was drawn.
        current_point (tuple): Coordinates of the current point where the line is being drawn.
        canvas (numpy.ndarray): A numpy array representing the canvas image.
        color (tuple, optional): Color of the line. Defaults to (0, 0, 255) (red).
        thickness (int, optional): Thickness of the line. Defaults to 2.
        distance_threshold (int, optional): Maximum distance between points to draw a line. Defaults to 160.

    Returns:
        tuple: Coordinates of the current point.
    """
    if last_point is not None and np.linalg.norm(np.array(last_point) - np.array(current_point)) < distance_threshold: # and is_registered(color):
        cv2.line(canvas, last_point, current_point, color, thickness, lineType=cv2.LINE_AA)
    return current_point

def skew_point(point, skew_matrix):
    """
    Skews a 2D point using a skew matrix.

    Parameters:
        point (tuple): Coordinates of the point to be skewed.
        skew_matrix (numpy.ndarray): 2x3 skew matrix for skew transformation.

    Returns:
        tuple: Coordinates of the skewed point.
    """
    homogeneous_point = np.array([point[0], point[1], 1])
    skewed_point = np.dot(skew_matrix, homogeneous_point)
    return (int(skewed_point[0] / skewed_point[2]), int(skewed_point[1] / skewed_point[2]))