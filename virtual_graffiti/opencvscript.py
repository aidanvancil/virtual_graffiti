'''
import cv2
import numpy as np

def enumerate_cameras():
    index = 0
    while True:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.read()[0]:
            break
        cap.release()
        index += 1

    return index


if __name__ == "__main__":
    camera_index = enumerate_cameras()

    if camera_index == 0:
        print("No cameras found.")
    else:
        print(f"Number of available cameras: {camera_index}")

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        # Define color ranges for red and green lasers (adjust these based on your laser colors)
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])
        lower_green = np.array([40, 100, 100])
        upper_green = np.array([80, 255, 255])

        # Set up canvas window
        canvas_width, canvas_height = 800, 600
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        canvas_window_name = 'Canvas'
        cv2.namedWindow(canvas_window_name)

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error: Failed to capture frame.")
                break


            # Display the original frame, red laser edges, green laser edges, and canvas
            cv2.imshow('Original', frame)
            cv2.imshow(canvas_window_name, canvas)

            # Break the loop if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        
'''
import cv2
import numpy as np
from screeninfo import get_monitors
import random
import math

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

    cap = cv2.VideoCapture(camera_indexes[1], cv2.CAP_DSHOW) 
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

                        if mode == 'fill' and background_image is not None:
                            update_canvas_with_image(canvas, background_image, cx, cy, scale_factor)


                            filled_pixels, total_pixels = count_filled_pixels(canvas, background_image)
                            fill_percentage = filled_pixels / total_pixels


                            if fill_percentage >= FILL_THRESHOLD_PERCENT:
                                # Apply glitter effect before filling the entire image
                                apply_glitter_effect(canvas, background_image)
                                # Fill in the entire image
                                canvas[:, :] = background_image[:, :]
                        elif mode == 'free':
                            color = (0, 0, 255) if color_index == 0 else (0, 255, 0)
                            if color_index == 0:
                                last_point_red = smooth_drawing(last_point_red, current_point, canvas, color)
                            else:
                                last_point_green = smooth_drawing(last_point_green, current_point, canvas, color)

        cv2.imshow('Original', frame)
        cv2.imshow(canvas_window_name, canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()