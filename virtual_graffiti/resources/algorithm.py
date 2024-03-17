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
        if not cap.read()[0]:
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
    segmented = cv2.bitwise_and(frame, frame, mask=mask)
    segmented_gray = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)
    return segmented_gray

def load_scaled_image(image_path, width, height):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Image not found")
    return cv2.resize(image, (width, height))

#UC04
def update_canvas_with_image(canvas, background_image, x, y, scale_factor, radius=25):
    scaled_x = int(x * scale_factor)
    scaled_y = int(y * scale_factor)
    mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (scaled_x, scaled_y), radius, 255, -1)
    mask_indices = np.where(mask > 0)
    canvas[mask_indices] = background_image[mask_indices]

def smooth_drawing(last_point, current_point, canvas, color=(0, 0, 255), thickness=2, distance_threshold=50):
    if last_point is not None and calculate_distance(last_point, current_point) < distance_threshold:
        cv2.line(canvas, last_point, current_point, color, thickness)
    return current_point

def count_filled_pixels(canvas, background_image):
    filled_pixels = np.sum(np.any(canvas != [0, 0, 0], axis=2))
    total_pixels = background_image.shape[0] * background_image.shape[1]
    return filled_pixels, total_pixels

#UC11
def apply_glitter_effect(canvas, canvas_window_name, background_image, iterations=400, intensity=600, delay=8):
    for _ in range(iterations):
        for _ in range(intensity):
            x, y = random.randint(0, canvas.shape[1] - 1), random.randint(0, canvas.shape[0] - 1)
            if np.all(canvas[y, x] == [0, 0, 0]): 
                canvas[y, x] = background_image[y, x]
        cv2.imshow(canvas_window_name, canvas)
        cv2.waitKey(delay)

#FR1, UC10
def init():
    data_queue = PriorityQueue()
    camera_indexes = enumerate_cameras()
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

    FILL_THRESHOLD_PERCENT = 0.80
    HOST = 'localhost'
    PORT = 9999

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
            print(curr_image)
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

                            if mode == 'fill' and background_image is not None:
                                update_canvas_with_image(canvas, background_image, cx, cy, scale_factor)

                                filled_pixels, total_pixels = count_filled_pixels(canvas, background_image)
                                fill_percentage = filled_pixels / total_pixels

                                if fill_percentage >= FILL_THRESHOLD_PERCENT:
                                    # Apply glitter effect before filling the entire image
                                    apply_glitter_effect(canvas, canvas_window_name, background_image)
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
                absolute_path = os.path.abspath('./../virtual_graffiti/virtual_graffiti/temp/reset_signal.txt')
                try:
                    with open(absolute_path, 'w') as f:
                            f.seek(0)
                            f.write('1')
                except Exception as e:
                    print(e)
                break
        conn.close()
            

    cap.release()
    cv2.destroyAllWindows()
        
if __name__ == "__main__":
    init()