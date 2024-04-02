from django.shortcuts import render, redirect, get_object_or_404
from django.http import StreamingHttpResponse, JsonResponse, HttpResponse
from screeninfo import get_monitors
import random
import cv2
import threading
import numpy as np
from queue import PriorityQueue
import importlib
import math
import os 
import time
import socket
import threading

def handle_client_connection(conn, data_queue):
    while True:
        data = conn.recv(1024).decode()
        if not data:
            break
        print("queued data:", data)
        if data == 'pull':
            data_queue.put((0, data))
        else:
            data_queue.put((1, data))

def enumerate_cameras():
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            break
        arr.append(index)
        cap.release()
        index += 1
    return arr

def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

#NFR6
def is_laser_contour(contour, hsv_frame, min_area=20, max_area=200):
    area = cv2.contourArea(contour)
    return min_area < area < max_area

def color_segmentation(frame, lower_color, upper_color):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    return mask

def load_scaled_image(image_path, width, height):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'app'))
    image_path = os.path.join(base_dir, image_path.lstrip('/'))
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Image not found")
    return cv2.resize(image, (width, height))

#UC04
def update_canvas_with_image(canvas, background_image, x, y, scale_factor, radius=140):
    scaled_x = int(x * scale_factor)
    scaled_y = int(y * scale_factor)
    mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (scaled_x, scaled_y), radius, 255, -1)
    mask_indices = np.where(mask > 0)
    canvas[mask_indices] = background_image[mask_indices]

# def is_registered(color):
#     red = (0, 0, 255)
#     green = (0, 255, 0)
#     color_map = {
#         str(green): (None, None),
#         str(red): (None, None)
#     }
#     try:
#         laser = Laser.objects.get(color=color_map[color])
#     except Laser.DoesNotExist:
#         return False

#     try:
#         UserProfile.objects.get(laser=laser)
#         return True
#     except UserProfile.DoesNotExist:
#         return False

def smooth_drawing(last_point, current_point, canvas, color=(0, 0, 255), thickness=2, distance_threshold=50):
    if last_point is not None and calculate_distance(last_point, current_point) < distance_threshold: # and is_registered(color):
        cv2.line(canvas, last_point, current_point, color, thickness)
    return current_point

def count_filled_pixels(canvas, background_image):
    filled_pixels = np.sum(np.any(canvas != [0, 0, 0], axis=2))
    total_pixels = background_image.shape[0] * background_image.shape[1]
    return filled_pixels, total_pixels

def calculate_thickness(dist):
    thickness = 24
    if 12 <= dist < 25:
        thickness = 22
    elif 25 <= dist < 37:
        thickness = 20
    elif 37 <= dist < 49:
        thickness = 18
    elif 49 <= dist < 61:
        thickness = 16
    elif 61 <= dist < 73:
        thickness = 14
    elif 73 <= dist < 85:
        thickness = 12
    elif 85 <= dist < 97:
        thickness = 10
    elif 97 <= dist < 109:
        thickness = 8
    elif 109 <= dist < 121:
        thickness = 6
    elif 121 <= dist < 133:
        thickness = 4
    elif 133 <= dist:
        thickness = 2
    return thickness

#UC11
def apply_glitter_effect(canvas, canvas_window_name, background_image, iterations=400, intensity=600, delay=8):
    for _ in range(iterations):
        for _ in range(intensity):
            x, y = random.randint(0, canvas.shape[1] - 1), random.randint(0, canvas.shape[0] - 1)
            if np.all(canvas[y, x] == [0, 0, 0]): 
                canvas[y, x] = background_image[y, x]
        cv2.imshow(canvas_window_name, canvas)
        cv2.waitKey(delay)


def clear_canvas(canvas):
    canvas[:, :] = 0

#FR1, UC10
def init():
    data_queue = PriorityQueue()
    camera_indexes = enumerate_cameras()
    print(camera_indexes)
    if len(camera_indexes) == 0:
        print('No cameras found')
        return
    print('Camera found')

    red_lower = np.array([160, 100, 100])
    red_upper = np.array([190, 255, 255])
    green_lower = np.array([50, 100, 100])
    green_upper = np.array([70, 255, 255])
    
    cap_idx = camera_indexes[0]
    cap = cv2.VideoCapture(cap_idx, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
    screen_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    screen_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    canvas_width, canvas_height = int(screen_width), int(screen_height)
    scale_factor = min(canvas_width / screen_width, canvas_height / screen_height)
    
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    canvas_window_name = 'Canvas'
    
    cv2.namedWindow(canvas_window_name, cv2.WINDOW_NORMAL)
    last_point_red = None
    last_point_green = None
    
    monitors = get_monitors()
    if len(monitors) > 1:
        external_monitor = monitors[1]
        cv2.moveWindow(canvas_window_name, external_monitor.x, external_monitor.y)

    
    cv2.setWindowProperty(canvas_window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    frame_cnt = 0
    FILL_THRESHOLD_PERCENT = .75
    FRAME_DIVISOR = 5
    MAX_DIST = 1024 
    HOST = 'localhost'
    PORT = 9999
    curr_image = None
    clear_area_center = (50, 50)  
    clear_area_size = (50, 50) 
    clear_area_rect = [clear_area_center[0] - clear_area_size[0] // 2, clear_area_center[1] - clear_area_size[1] // 2, clear_area_size[0], clear_area_size[1]]
    clear_area_start_time = None
    red = (0, 0, 255)
    green = (0, 255, 0)
    prev = {
        str(green): (None, None),
        str(red): (None, None)
    }

    prev_thickness = {
        str(green): 2,
        str(red):  2
    }

    #UC03
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
        server_sock.bind((HOST, PORT))
        server_sock.listen()
        server_sock.setblocking(False)

        print("Waiting for connection...")
        while True:
            
            try:
                conn, addr = server_sock.accept()
                print("Connected by", addr)
                client_thread = threading.Thread(target=handle_client_connection, args=(conn, data_queue))
                client_thread.start()

            except BlockingIOError:
                pass

            except Exception as e:
                print("Error:", e)
            
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            if not data_queue.empty() and data_queue.queue[0][1] == 'pull':
                data_queue.get()
                if not data_queue.empty():
                    received_data = data_queue.get()[1]
                    curr_image = received_data
                    
            mode = 'fill' if curr_image else 'free'
            background_image = None
            if mode == 'fill':
                if curr_image:
                    background_image = load_scaled_image(curr_image, canvas_width, canvas_height)
                    background_image = background_image[:, :, :3]
                else:
                    JsonResponse({'message': 'Image not loaded successfully.'}, status=405)

            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            red_segmented = color_segmentation(frame, red_lower, red_upper)
            green_segmented = color_segmentation(frame, green_lower, green_upper)
            #purple_segmented = color_segmentation(frame, purple_lower, purple_upper)

            # Process for both rgb lasers in both modes
            for color_index, segmented in enumerate([red_segmented, green_segmented]): #, purple_segmented]):
                contours, _ = cv2.findContours(segmented, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    if is_laser_contour(contour, hsv_frame):
                        moments = cv2.moments(contour)
                        if moments["m00"] != 0:
                            cx = int(moments["m10"] / moments["m00"])
                            cy = int(moments["m01"] / moments["m00"])
                            current_point = (cx, cy)

                            if clear_area_rect[0] <= cx <= clear_area_rect[0] + clear_area_rect[2] and clear_area_rect[1] <= cy <= clear_area_rect[1] + clear_area_rect[3]:
                                if clear_area_start_time is None:
                                    clear_area_start_time = time.time()
                                elif time.time() - clear_area_start_time >= 1.5:  # Laser has been in the clear area for 3 seconds
                                    clear_canvas(canvas)
                                    clear_area_start_time = None  # Reset the timer
                            else:
                                clear_area_start_time = None  # Reset the timer if the laser moves out of the clear area

                            if mode == 'fill' and background_image is not None:
                                update_canvas_with_image(canvas, background_image, cx, cy, scale_factor)

                                filled_pixels, total_pixels = count_filled_pixels(canvas, background_image)
                                fill_percentage = filled_pixels / total_pixels

                                if fill_percentage >= FILL_THRESHOLD_PERCENT:
                                    # Apply glitter effect before filling the entire image
                                    apply_glitter_effect(canvas, canvas_window_name, background_image)
                                    canvas[:, :] = background_image[:, :]
                                    time.sleep(5)
                                    curr_image = None
                            elif mode == 'free':
                                color = red if color_index == 0 else green
                                if frame_cnt % FRAME_DIVISOR == 0:
                                    (x, y) = prev[str(color)]
                                    if x is None or y is None:
                                        if color_index == 0:
                                            last_point_red = smooth_drawing(last_point_red, current_point, canvas, color)
                                        else:
                                            last_point_green = smooth_drawing(last_point_green, current_point, canvas, color)
                                        prev[str(color)] = (cx, cy)
                                        continue
                                        
                                    dist = calculate_distance((cx, cy), prev[str(color)])
                                    prev[str(color)] = (cx, cy)
                                    if MAX_DIST - dist <= 0:
                                        dist = 0
                                    thickness = calculate_thickness(dist)
                                    prev_thickness[str(color)] = int(thickness)

                                if color_index == 0:
                                    last_point_red = smooth_drawing(last_point_red, current_point, canvas, color, thickness=prev_thickness[str(color)])
                                else:
                                    last_point_green = smooth_drawing(last_point_green, current_point, canvas, color, thickness=prev_thickness[str(color)])

            frame_cnt += 1
                                

            
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
                absolute_path = os.path.abspath('./../virtual_graffiti/virtual_graffiti/temp/reset_signal.txt')
                try:
                    with open(absolute_path, 'w') as f:
                            f.seek(0)
                            f.write('1')
                except Exception as e:
                    print(e)
                break
            elif key == ord('c'):  # Clear canvas if 'c' is pressed
                clear_canvas(canvas)
        conn.close()
            

    cap.release()
    cv2.destroyAllWindows()
        
if __name__ == "__main__":
    init()