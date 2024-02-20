from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login as log, logout as auth_logout
from django.http import StreamingHttpResponse, JsonResponse
from django.views.decorators import gzip
from django.conf import settings as _settings

from app.models import UserProfile, Laser
from screeninfo import get_monitors
import random
import cv2
import qrcode
import base64
import os
from io import BytesIO
import json
import numpy as np

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

def is_laser_contour(contour, hsv_frame, min_area=20, max_area=200):
    if not min_area < cv2.contourArea(contour) < max_area:
        return False
    return True

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

def update_canvas_with_image(canvas, background_image, x, y, scale_factor, radius=5):
    scaled_x = int(x * scale_factor)
    scaled_y = int(y * scale_factor)
    mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (scaled_x, scaled_y), radius, 255, -1)
    mask_indices = np.where(mask > 0)
    canvas[mask_indices] = background_image[mask_indices]

def count_filled_pixels(canvas, background_image):
    # Count pixels that are not black (0, 0, 0) in the canvas
    filled_pixels = np.sum(np.any(canvas != [0, 0, 0], axis=2))
    total_pixels = background_image.shape[0] * background_image.shape[1]
    return filled_pixels, total_pixels

def apply_glitter_effect(canvas, canvas_window_name, background_image, iterations=400, intensity=600, delay=8):
    for _ in range(iterations):
        for _ in range(intensity):
            x, y = random.randint(0, canvas.shape[1] - 1), random.randint(0, canvas.shape[0] - 1)
            if np.all(canvas[y, x] == [0, 0, 0]): 
                canvas[y, x] = background_image[y, x]
        cv2.imshow(canvas_window_name, canvas)
        cv2.waitKey(delay)

def submit_image(request):
    if request.method == 'POST':
        image_queue = request.session.get('image_queue', [])
        curr_image = request.POST.get('image_url')
        image_queue.append(curr_image)
        request.session['image_queue'] = image_queue
        return JsonResponse({'message': 'Image submitted successfully.'}, status=200)
    else:
        print(request)
        return JsonResponse({'error': 'Invalid request method.'}, status=405)
    
def poll(request):
    if request.method == 'GET':    
        if not request.session.get('init', False):
            request.session['init'] = True

        red_lower = np.array([0, 100, 100])
        red_upper = np.array([10, 255, 255])
        green_lower = np.array([40, 100, 100])
        green_upper = np.array([80, 255, 255])
        purple_upper = np.array([130, 50, 50])
        purple_lower = np.array([160, 255, 255])

        image_queue = request.session.get('image_queue', [])
        curr_image = None
        if image_queue:
            curr_image = image_queue.pop(0)
            
        mode = 'fill' if curr_image else 'free'
        camera_indexes = enumerate_cameras()
        
        if len(camera_indexes) == 0:
            return JsonResponse({'message': 'No cameras found.'}, status=400)

        cap = cv2.VideoCapture(camera_indexes[0], cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FPS, 60)
        screen_width = cap.get(3) 
        screen_height = cap.get(4)
        
        canvas_width, canvas_height = int(screen_width), int(screen_height)
        
        scale_factor_x = canvas_width / screen_width
        scale_factor_y = canvas_height / screen_height
        scale_factor = min(scale_factor_x, scale_factor_y)

        background_image = None
        if mode == 'fill':
            if curr_image:
                background_image = load_scaled_image(curr_image, canvas_width, canvas_height)
                background_image = background_image[:, :, :3]
            else:
                JsonResponse({'message': 'Image not loaded successfully.'}, status=405)

        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        canvas_window_name = 'Canvas'
        
        cv2.namedWindow(canvas_window_name, cv2.WINDOW_NORMAL)
        
        
        # Detect if an external monitor is present and adjust window position
        monitors = get_monitors()
        if len(monitors) > 1:
            # Assume you want to display on the second monitor (external monitor)
            external_monitor = monitors[1]
            cv2.moveWindow(canvas_window_name, external_monitor.x, external_monitor.y)

        
        cv2.setWindowProperty(canvas_window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        FILL_THRESHOLD_PERCENT = 0.80

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            red_segmented = color_segmentation(frame, red_lower, red_upper)
            green_segmented = color_segmentation(frame, green_lower, green_upper)
            purple_segmented = color_segmentation(frame, purple_lower, purple_upper)

            # Process for both rgb lasers in both modes
            for color_index, segmented in enumerate([red_segmented, green_segmented, purple_segmented]):
                contours, _ = cv2.findContours(segmented, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    if is_laser_contour(contour, hsv_frame):
                        moments = cv2.moments(contour)
                        if moments["m00"] != 0:
                            cx = int(moments["m10"] / moments["m00"])
                            cy = int(moments["m01"] / moments["m00"])

                            if mode == 'fill' and background_image is not None:
                                update_canvas_with_image(canvas, background_image, cx, cy, scale_factor)

                                filled_pixels, total_pixels = count_filled_pixels(canvas, background_image)
                                fill_percentage = filled_pixels / total_pixels

                                if fill_percentage >= FILL_THRESHOLD_PERCENT:
                                    # Apply glitter effect before filling the entire image
                                    apply_glitter_effect(canvas, canvas_window_name, background_image)
                                    # Fill in the entire image
                                    canvas[:, :] = background_image[:, :]
                                    request.session['curr_image'] = None

                            elif mode == 'free':
                                color = (0, 0, 255) if color_index == 0 else (0, 255, 0)
                                cv2.circle(canvas, (cx, cy), 2, color, -1)

            cv2.imshow('Original', frame)
            cv2.imshow(canvas_window_name, canvas)
                    
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

@gzip.gzip_page
def video_feed(request):
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    def generate():
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
            
                _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                frame_bytes = jpeg.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
        finally:
            cap.release()

    response = StreamingHttpResponse(generate(), content_type="multipart/x-mixed-replace;boundary=frame")
    return response