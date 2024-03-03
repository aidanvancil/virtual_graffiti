from django.shortcuts import render, redirect
from django.views.decorators import gzip
from django.http import StreamingHttpResponse, JsonResponse, HttpResponse
from subprocess import Popen
from channels.layers import get_channel_layer
import cv2
import os

def submit_image(request):
    if request.method == 'POST':
        channel_layer = get_channel_layer()
        image_url = request.POST.get('image_url')
        channel_layer.group_send('myconsumergroup', {'type': 'send.image.url', 'image_url': image_url})
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

def poll(request):
    return render(request, 'admin_panel.html')

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