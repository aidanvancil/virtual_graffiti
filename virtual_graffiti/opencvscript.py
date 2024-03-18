import cv2
import numpy as np
from screeninfo import get_monitors
import random
import math
import time

# Optimized Euclidean distance calculation
def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def enumerate_cameras():
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.read()[0]:
            break
        arr.append(index)
        cap.release()
        index += 1
    return arr

# Adjusted for improved performance
def is_laser_contour(contour, hsv_frame, min_area=20, max_area=200):
    area = cv2.contourArea(contour)
    return min_area < area < max_area

# Optimized color segmentation to reduce computational load
def color_segmentation(frame, lower_color, upper_color):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    return mask

def load_scaled_image(image_path, width, height):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Image not found")
    return cv2.resize(image, (width, height))

def update_canvas_with_image(canvas, background_image, x, y, scale_factor, radius=50):
    scaled_x = int(x * scale_factor)
    scaled_y = int(y * scale_factor)
    mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (scaled_x, scaled_y), radius, 255, -1)
    mask_indices = np.where(mask > 0)
    canvas[mask_indices] = background_image[mask_indices]

# Improved drawing smoothing
def smooth_drawing(last_point, current_point, canvas, color=(0, 0, 255), thickness=2, distance_threshold=50):
    if last_point is not None and calculate_distance(last_point, current_point) < distance_threshold:
        cv2.line(canvas, last_point, current_point, color, thickness)
    return current_point

def count_filled_pixels(canvas, background_image):
    filled_pixels = np.sum(np.any(canvas != [0, 0, 0], axis=2))
    total_pixels = background_image.shape[0] * background_image.shape[1]
    return filled_pixels, total_pixels

def apply_glitter_effect(canvas, background_image, iterations=400, intensity=600, delay=8):
    for _ in range(iterations):
        for _ in range(intensity):
            x, y = random.randint(0, canvas.shape[1] - 1), random.randint(0, canvas.shape[0] - 1)
            if np.all(canvas[y, x] == [0, 0, 0]):
                canvas[y, x] = background_image[y, x]
        cv2.imshow(canvas_window_name, canvas)
        cv2.waitKey(delay)

def clear_canvas(canvas):
    canvas[:, :] = 0

if __name__ == "__main__":
    red_lower = np.array([160, 100, 100])
    red_upper = np.array([190, 255, 255])
    green_lower = np.array([50, 100, 100])
    green_upper = np.array([70, 255, 255])

    mode = input("Choose mode (fill/free): ").strip().lower()
    if mode not in ['fill', 'free']:
        print("Invalid mode selected. Exiting.")
        exit()

    camera_indexes = enumerate_cameras()
    if len(camera_indexes) == 0:
        print("No cameras found.")
        exit()

    # Use the most suitable camera index for your setup
    cap = cv2.VideoCapture(camera_indexes[0], cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
    screen_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    screen_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    canvas_width, canvas_height = int(screen_width), int(screen_height)
    scale_factor = min(canvas_width / screen_width, canvas_height / screen_height)
    background_image = None
    if mode == 'fill':
        background_image = load_scaled_image(r"C:\Users\moren\Desktop\moshe\daftpunkwallpaper.jpg", canvas_width, canvas_height)

    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    canvas_window_name = 'Canvas'
    cv2.namedWindow(canvas_window_name, cv2.WINDOW_NORMAL)
    monitors = get_monitors()
    if len(monitors) > 1:
        external_monitor = monitors[1]
        cv2.moveWindow(canvas_window_name, external_monitor.x, external_monitor.y)

    cv2.setWindowProperty(canvas_window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    last_point_red = None
    last_point_green = None

    FILL_THRESHOLD_PERCENT = .75

    # Define clear area parameters
    clear_area_center = (50, 50)  
    clear_area_size = (50, 50) 
    clear_area_rect = [clear_area_center[0] - clear_area_size[0] // 2, clear_area_center[1] - clear_area_size[1] // 2, clear_area_size[0], clear_area_size[1]]
    clear_area_start_time = None  # To track how long the laser stays within the clear area

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        red_segmented = color_segmentation(frame, red_lower, red_upper)
        green_segmented = color_segmentation(frame, green_lower, green_upper)

        for color_index, segmented in enumerate([red_segmented, green_segmented]):
            contours, _ = cv2.findContours(segmented, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if is_laser_contour(contour, frame):
                    moments = cv2.moments(contour)
                    if moments["m00"] != 0:
                        cx = int(moments["m10"] / moments["m00"])
                        cy = int(moments["m01"] / moments["m00"])
                        current_point = (cx, cy)

                        # Check if current_point is within the clear area
                        if clear_area_rect[0] <= cx <= clear_area_rect[0] + clear_area_rect[2] and clear_area_rect[1] <= cy <= clear_area_rect[1] + clear_area_rect[3]:
                            if clear_area_start_time is None:
                                clear_area_start_time = time.time()
                            elif time.time() - clear_area_start_time >= 3:  # Laser has been in the clear area for 3 seconds
                                clear_canvas(canvas)
                                clear_area_start_time = None  # Reset the timer
                        else:
                            clear_area_start_time = None  # Reset the timer if the laser moves out of the clear area

                        if mode == 'fill' and background_image is not None:
                            update_canvas_with_image(canvas, background_image, cx, cy, scale_factor)
                            filled_pixels, total_pixels = count_filled_pixels(canvas, background_image)
                            fill_percentage = filled_pixels / total_pixels
                            if fill_percentage >= FILL_THRESHOLD_PERCENT:
                                apply_glitter_effect(canvas, background_image)
                                canvas[:, :] = background_image[:, :]
                        elif mode == 'free':
                            color = (0, 0, 255) if color_index == 0 else (0, 255, 0)
                            if color_index == 0:
                                last_point_red = smooth_drawing(last_point_red, current_point, canvas, color)
                            else:
                                last_point_green = smooth_drawing(last_point_green, current_point, canvas, color)

        # Draw the clear area for visual feedback
        cv2.rectangle(canvas, (clear_area_rect[0], clear_area_rect[1]), (clear_area_rect[0] + clear_area_rect[2], clear_area_rect[1] + clear_area_rect[3]), (255, 255, 255), 2)
                
        # Add "clear" text inside the clear area
        font_scale = 0.5
        font_thickness = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize("clear", font, font_scale, font_thickness)[0]
        text_x = clear_area_rect[0] + (clear_area_rect[2] - text_size[0]) // 2
        text_y = clear_area_rect[1] + (clear_area_rect[3] + text_size[1]) // 2
        cv2.putText(canvas, "clear", (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

        cv2.imshow('Original', frame)
        cv2.imshow(canvas_window_name, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):  # Clear canvas if 'c' is pressed
            clear_canvas(canvas)

    cap.release()
    cv2.destroyAllWindows()
