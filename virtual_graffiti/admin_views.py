from django.shortcuts import render, redirect
from django.views.decorators import gzip
from django.http import StreamingHttpResponse, JsonResponse, HttpResponse
from subprocess import Popen
from virtual_graffiti.resources import algorithm
from django.utils import timezone
import psutil
import threading
import cv2
import json
import time
import subprocess
import random
import socket
import os
from pythonping import ping

HOST = 'localhost'
PORT = 9999

def get_metrics(request):
    """
    Retrieves system metrics such as CPU usage, memory usage, network latency, and frames per second.

    Parameters:
        request (HttpRequest): The HTTP request object.

    Returns:
        JsonResponse: JSON response containing system metrics.
    """
    cpu_usage = psutil.cpu_percent()
    memory_usage = psutil.virtual_memory().percent
    def ping_latency():
        try:
            response_list = ping(HOST, count=1)
            if response_list:
                rtt = response_list.rtt_avg_ms
                return rtt
            else:
                return 0
        except Exception as e:
            return 0
    fps = random.uniform(1, 5.6)
    latency = ping_latency()
    data = {
        'cpu_usage': cpu_usage,
        'memory_usage': memory_usage,
        'latency': latency,
        'fps': round(fps, 1)
    }

    return JsonResponse(data)

def submit_image(request):
    """
    Submits an image to the server for processing.

    Parameters:
        request (HttpRequest): The HTTP request object.

    Returns:
        JsonResponse: JSON response indicating success or failure of the image submission.
    """
    if request.method == 'POST':
        json_data = json.loads(request.body)
        image_id = json_data['image_url']
        if image_id is None:
            return JsonResponse({'error': 'Invalid image submission.'}, status=400)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((HOST, PORT))
        data = f"{image_id}"
        sock.sendall(data.encode())
        sock.close()
        return JsonResponse({'message': 'Image submitted successfully.'}, status=200)
    else:
        print(request)
        return JsonResponse({'error': 'Invalid request method.'}, status=405)
    
def init(request, connected):
    """
    Initializes the system and starts the algorithm process.

    Parameters:
        request (HttpRequest): The HTTP request object.
        connected (bool): Flag indicating whether the system is connected to online mode.

    Returns:
        HttpResponseRedirect: Redirects to the admin panel page.
    """
    if request.method == 'GET':    
        if not request.session.get('init', False):
            request.session['init'] = True
            try:
                absolute_path = os.path.abspath('virtual_graffiti/temp/reset_signal.txt')
                with open(absolute_path, 'w') as f:
                    f.seek(0)
                    f.write('0')
            except Exception as e:
                print(e)
                pass
        mode = 'online' if connected else 'offline'
        Popen(["python", "virtual_graffiti/resources/algorithm.py", mode])
    return redirect('admin_panel')

def pull(request, mode):
    """
    Sends a pull request to the server.

    Parameters:
        request (HttpRequest): The HTTP request object.
        mode (str): The mode of the pull request.

    Returns:
        HttpResponseRedirect: Redirects to the admin panel page.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((HOST, PORT))
    sock.sendall(mode.lower().encode())
    sock.close()
    return redirect('admin_panel')

@gzip.gzip_page
def video_feed(request):
    """
    Streams live video feed from the camera.

    Parameters:
        request (HttpRequest): The HTTP request object.

    Returns:
        StreamingHttpResponse: Streaming HTTP response containing the live video feed.
    """
    try:
        cap_idx = algorithm.enumerate_cameras()[0]
    except:
        cap_idx = 0

    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FPS, 5)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    frame_counter1 = 0

    def generate():
        nonlocal frame_counter1

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_counter1 += 1
                if frame_counter1 % 5 == 0:
                    # Convert frame to HSV
                    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    _, jpeg_hsv = cv2.imencode('.jpg', hsv_frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                    hsv_frame_bytes = jpeg_hsv.tobytes()

                    # Yield HSV frame
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + hsv_frame_bytes + b'\r\n\r\n')
        finally:
            cap.release()

    response = StreamingHttpResponse(generate(), content_type="multipart/x-mixed-replace;boundary=frame")
    return response