import random
import cv2
import numpy as np
import os 
from .computing import skew_point

def clear_canvas(canvas):
    """
    Clears the canvas by setting all pixels to black.

    Parameters:
        canvas (numpy.ndarray): The canvas image represented as a NumPy array.

    Returns:
        None
    """
    canvas[:, :] = 0
    
def find_black_coordinates(canvas):
    """
    Finds the coordinates of black pixels (0, 0, 0) in the canvas image.

    Parameters:
        canvas (numpy.ndarray): The canvas image represented as a NumPy array.

    Returns:
        numpy.ndarray: Array of (x, y) coordinates of black pixels.
    """
    black_coords = np.argwhere(np.all(canvas == [0, 0, 0], axis=-1))
    return black_coords

def load_scaled_image(image_path, width, height):
    """
    Loads an image from the specified path and resizes it to the given width and height.

    Parameters:
        image_path (str): The path to the image file.
        width (int): The target width of the image.
        height (int): The target height of the image.

    Returns:
        numpy.ndarray: The loaded and resized image.
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'app'))
    image_path = os.path.join(base_dir, image_path.lstrip('/'))
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Image not found")
    return cv2.resize(image, (width, height))

def update_canvas_with_image(canvas, background_image, x, y, scale_factor, radius=180):
    """
    Updates the canvas with a background image centered at the specified location.

    Parameters:
        canvas (numpy.ndarray): The canvas image represented as a NumPy array.
        background_image (numpy.ndarray): The background image to be overlaid on the canvas.
        x (int): The x-coordinate of the center of the background image.
        y (int): The y-coordinate of the center of the background image.
        scale_factor (float): The scale factor to resize the image.
        radius (int): The radius of the circular mask for overlaying the image.

    Returns:
        None
    """
    scaled_x = int(x * scale_factor)
    scaled_y = int(y * scale_factor)
    mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (scaled_x, scaled_y), radius, 255, -1)
    mask_indices = np.where(mask > 0)
    canvas[mask_indices] = background_image[mask_indices]

def apply_glitter_effect(canvas, background_image, iterations=400, intensity=10000, delay=2):
    """
    Applies a glitter effect to the canvas by randomly replacing black pixels with pixels from the background image.

    Parameters:
        canvas (numpy.ndarray): The canvas image represented as a NumPy array.
        background_image (numpy.ndarray): The background image to sample pixels from.
        iterations (int): The number of iterations for the glitter effect.
        intensity (int): The intensity of the glitter effect.
        delay (int): The delay between each iteration in milliseconds.

    Returns:
        None
    """
    black_coords = find_black_coordinates(canvas)
    for _ in range(iterations):
        for _ in range(intensity):
            index = random.randint(0, len(black_coords) - 1)
            y, x = black_coords[index]
            canvas[y, x] = background_image[y, x]
        cv2.imshow('Canvas', canvas)
        cv2.waitKey(delay)

def get_clear_button(skew_matrix, clear_area_rect):
    """
    Transforms the coordinates of a clear button rectangle using a skew matrix.

    Parameters:
        skew_matrix (numpy.ndarray): The skew matrix for transforming coordinates.
        clear_area_rect (list): List of four (x, y) coordinates representing the clear button rectangle.

    Returns:
        list: List of four transformed (x, y) coordinates representing the skewed clear button rectangle.
    """
    skewed_clear_area_rect = [skew_point(clear_area_rect[0], skew_matrix),
                              skew_point(clear_area_rect[1], skew_matrix),
                              skew_point(clear_area_rect[3], skew_matrix),
                              skew_point(clear_area_rect[2], skew_matrix),
                            ]
    return skewed_clear_area_rect

def get_color_palette(skew_matrix, color_pallete_rect):
    """
    Transforms the coordinates of a color palette rectangle using a skew matrix.

    Parameters:
        skew_matrix (numpy.ndarray): The skew matrix for transforming coordinates.
        color_pallete_rect (list): List of four (x, y) coordinates representing the color palette rectangle.

    Returns:
        list: List of four transformed (x, y) coordinates representing the skewed color palette rectangle.
    """
    skewed_clear_area_rect = [skew_point(color_pallete_rect[0], skew_matrix),
                              skew_point(color_pallete_rect[1], skew_matrix),
                              skew_point(color_pallete_rect[3], skew_matrix),
                              skew_point(color_pallete_rect[2], skew_matrix),
                            ]
    return skewed_clear_area_rect

def draw_palette_box(skewed_canvas, palette_box, color):
    """
    Draws a filled polygon on the canvas representing the color palette box.

    Parameters:
        skewed_canvas (numpy.ndarray): The canvas image represented as a NumPy array.
        palette_box (list): List of four (x, y) coordinates representing the color palette box.
        color (tuple): Tuple representing the color of the palette box.

    Returns:
        numpy.ndarray: The canvas with the drawn palette box.
    """
    list_ver = [list(i) for i in palette_box]
    pts = np.array(list_ver, np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.fillPoly(skewed_canvas, [pts], color)
    return skewed_canvas

def draw_clear_button(skewed_canvas, skewed_clear_area_rect):
    """
    Draws a clear button on the canvas using a skewed rectangle.

    Parameters:
        skewed_canvas (numpy.ndarray): The canvas image represented as a NumPy array.
        skewed_clear_area_rect (list): List of four (x, y) coordinates representing the skewed clear button rectangle.

    Returns:
        numpy.ndarray: The canvas with the drawn clear button.
    """
    list_ver = [list(i) for i in skewed_clear_area_rect]
    pts = np.array(list_ver, np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.polylines(skewed_canvas, [pts], isClosed=True, color=(255, 255, 255), thickness=2)
    return skewed_canvas