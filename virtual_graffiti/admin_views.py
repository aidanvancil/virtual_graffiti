from django.shortcuts import render, redirect
from django.views.decorators import gzip
from django.http import StreamingHttpResponse, JsonResponse, HttpResponse
from subprocess import Popen
from virtual_graffiti.resources import algorithm
from django.utils import timezone
import threading
import cv2
import json
import socket
import os

def submit_image(request):
    if request.method == 'POST':
        json_data = json.loads(request.body)
        image_id = json_data['image_url']
        if image_id is None:
            return JsonResponse({'error': 'Invalid image submission.'}, status=400)
        
        HOST = 'localhost'
        PORT = 9999
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((HOST, PORT))
        data = f"{image_id}"
        sock.sendall(data.encode())
        sock.close()
        return JsonResponse({'message': 'Image submitted successfully.'}, status=200)
    else:
        print(request)
        return JsonResponse({'error': 'Invalid request method.'}, status=405)
    
def init(request):
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
            
        Popen(["python", "virtual_graffiti/resources/algorithm.py"])
    return redirect('admin_panel')

def pull(request):
    HOST = 'localhost'
    PORT = 9999
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((HOST, PORT))
    data = 'pull'
    sock.sendall(data.encode())
    sock.close()
    return redirect('admin_panel')

@gzip.gzip_page
def video_feed(request):
    cap = cv2.VideoCapture(2)
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