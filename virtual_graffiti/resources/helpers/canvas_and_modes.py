import random
import cv2
import numpy as np
import os 
from .computing import skew_point

def clear_canvas(canvas):
    canvas[:, :] = 0
    
def find_black_coordinates(canvas):
    black_coords = np.argwhere(np.all(canvas == [0, 0, 0], axis=-1))
    return black_coords

def load_scaled_image(image_path, width, height):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'app'))
    image_path = os.path.join(base_dir, image_path.lstrip('/'))
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Image not found")
    return cv2.resize(image, (width, height))

def update_canvas_with_image(canvas, background_image, x, y, scale_factor, radius=180):
    scaled_x = int(x * scale_factor)
    scaled_y = int(y * scale_factor)
    mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (scaled_x, scaled_y), radius, 255, -1)
    mask_indices = np.where(mask > 0)
    canvas[mask_indices] = background_image[mask_indices]

def apply_glitter_effect(canvas, background_image, iterations=400, intensity=10000, delay=2):
    black_coords = find_black_coordinates(canvas)
    for _ in range(iterations):
        for _ in range(intensity):
            index = random.randint(0, len(black_coords) - 1)
            y, x = black_coords[index]
            canvas[y, x] = background_image[y, x]
        cv2.imshow('Canvas', canvas)
        cv2.waitKey(delay)

def get_clear_button(skew_matrix, clear_area_rect):
    skewed_clear_area_rect = [skew_point(clear_area_rect[0], skew_matrix),
                              skew_point(clear_area_rect[1], skew_matrix),
                              skew_point(clear_area_rect[3], skew_matrix),
                              skew_point(clear_area_rect[2], skew_matrix),
                            ]
    return skewed_clear_area_rect

def get_color_palette(skew_matrix, color_pallete_rect):
    skewed_clear_area_rect = [skew_point(color_pallete_rect[0], skew_matrix),
                              skew_point(color_pallete_rect[1], skew_matrix),
                              skew_point(color_pallete_rect[3], skew_matrix),
                              skew_point(color_pallete_rect[2], skew_matrix),
                            ]
    return skewed_clear_area_rect

def draw_palette_box(skewed_canvas, palette_box, color):
    list_ver = [list(i) for i in palette_box]
    pts = np.array(list_ver, np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.fillPoly(skewed_canvas, [pts], color)
    return skewed_canvas

def draw_clear_button(skewed_canvas, skewed_clear_area_rect):
    list_ver = [list(i) for i in skewed_clear_area_rect]
    pts = np.array(list_ver, np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.polylines(skewed_canvas, [pts], isClosed=True, color=(255, 255, 255), thickness=2)
    return skewed_canvas