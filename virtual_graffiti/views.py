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

'''   
    Author(s): Foster Schmidt (F), Moises Moreno (M), Aidan Vancil (A)
    Date(s):   11/12/23 - 12/03/23
    
    Description:
    - (A + F) admin_panel, settings
    - (M + F) register, login
    - (A + M) video_feed
    - (M) errors
    - (A) set_laser_*, get_laser_*, disconnect
    - (F) logout
'''

HOST = "localhost:8000"

def get_laser(request, laser_id):
    if request.method == 'GET':
        try:
            laser = Laser.objects.get(id=laser_id)
        except Laser.DoesNotExist:
            return errors(request)

        print(laser)
        return JsonResponse({
            'color': laser.color,
            'size': laser.size,
            'style': laser.style
        })
        
    return JsonResponse({'success': False, 'error': 'Invalid request method'}, status=405)

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
    # Check the contour area
    if not min_area < cv2.contourArea(contour) < max_area:
        return False

    # Further checks can be added here (e.g., brightness, shape)
    # ...

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
            if np.all(canvas[y, x] == [0, 0, 0]):  # Check if the pixel is unfilled
                canvas[y, x] = background_image[y, x]
        cv2.imshow(canvas_window_name, canvas)
        cv2.waitKey(delay)

def initialize_projector(request):
    if request.method != 'GET':
        return JsonResponse({}, status=200)
    

    red_lower = np.array([0, 100, 100])
    red_upper = np.array([10, 255, 255])
    green_lower = np.array([40, 100, 100])
    green_upper = np.array([80, 255, 255])
    purple_upper = np.array([130, 50, 50])
    purple_lower = np.array([160, 255, 255])

    mode = print("Choose mode (fill/free): ")
    if mode not in ['fill', 'free']:
        print("Invalid mode selected. Exiting.")
    
    camera_indexes = enumerate_cameras()

    if len(camera_indexes) == 0:
        print("No cameras found.")
        exit()

    cap = cv2.VideoCapture(camera_indexes[0], cv2.CAP_DSHOW)

    screen_width = cap.get(3) 
    screen_height = cap.get(4)
    print(screen_height)
    print(screen_width) 
    
    canvas_width, canvas_height = int(screen_width), int(screen_height)
    
    scale_factor_x = canvas_width / screen_width
    scale_factor_y = canvas_height / screen_height
    scale_factor = min(scale_factor_x, scale_factor_y)

    background_image = None
    if mode == 'fill':
        background_image = load_scaled_image(r"C:\Users\aidan\Pictures\Screenshots\water.png", canvas_width, canvas_height)
        background_image = background_image[:, :, :3]

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
                        elif mode == 'free':
                            color = (0, 0, 255) if color_index == 0 else (0, 255, 0)
                            cv2.circle(canvas, (cx, cy), 2, color, -1)

        cv2.imshow('Original', frame)
        cv2.imshow(canvas_window_name, canvas)
        # Display the original frame and canvas
                
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

def set_laser_color(request, laser_id):
    data = json.loads(request.body)
    laser = Laser.objects.get(id=laser_id)
    laser.color = data['data']
    laser.save()
    return JsonResponse({'success': True}, status=200)

def set_laser_size(request, laser_id):
    data = json.loads(request.body)
    laser = Laser.objects.get(id=laser_id)
    laser.size = data['data']
    laser.save()
    return JsonResponse({'success': True}, status=200)

def set_laser_style(request, laser_id):
    data = json.loads(request.body)
    laser = Laser.objects.get(id=laser_id)
    laser.style = data['data']
    laser.save()
    return JsonResponse({'success': True}, status=200)
    
def errors(request):
    context = {
        'gradient': True,
        'from_gradient': '#74EE15',
        'to_gradient': '#F000FF',
        'error': 404
    }
    return render(request, 'errors.html', context)

def settings(request, user_identifier):
    try:
        user_identifier_decoded = base64.b64decode(user_identifier).decode('utf-8')
        first_name, last_name, laser_pointer = user_identifier_decoded.split('_')
    except:
        return errors(request)
    
    laser_pointer = Laser.objects.get(id=laser_pointer)
    user = UserProfile.objects.get(first_name=first_name, last_name=last_name, laser=laser_pointer)
    context = {
        'gradient': True,
        'from_gradient': '#74EE15',
        'to_gradient': '#F000FF',
        'first_name': user.first_name,
        'last_name': user.last_name,
        'laser_pointer': user.laser.id,
    }

    return render(request, 'settings.html', context)

@login_required(login_url='login')
def remove_user_and_release_laser(request, first_name, last_name):
    user_to_del = get_object_or_404(UserProfile, first_name=first_name, last_name=last_name)
    
    if user_to_del:
        user_to_del.delete()
    return redirect(admin_panel)

@login_required(login_url='login')
def signup(request):
    if request.method == 'POST':
        first_name = request.POST.get('firstname')
        last_name = request.POST.get('lastname')
        laser_pointer = request.POST.get('laser')
        laser = Laser.objects.get(id=laser_pointer)

        UserProfile.objects.create(first_name=first_name, last_name=last_name, laser=laser)
        base64_user_identifier = base64.b64encode(f"{first_name}_{last_name}_{laser_pointer}".encode('utf-8')).decode('utf-8')
        redirect_url = f"http://{HOST}/settings/{base64_user_identifier}"
       
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        
        qr.add_data(redirect_url)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")

        buffer = BytesIO()
        img.save(buffer)
        qr_code_image = buffer.getvalue()

        qr_code_base64 = base64.b64encode(qr_code_image).decode('utf-8')

        context = {
            'qr_code_base64': qr_code_base64,
            'gradient': True,
            'from_gradient': '#74EE15',
            'to_gradient': '#F000FF',
        }

        return render(request, 'signup.html', context)\
            
    else:  
        lasers_without_users = Laser.objects.filter(userprofile__isnull=True)
        context = {
            'gradient': True,
            'from_gradient': '#74EE15',
            'to_gradient': '#F000FF',
            'available_lasers': list(lasers_without_users.values_list('id', flat=True))
        }
        
    return render(request, 'signup.html', context)


def login(request):
    if request.user.is_authenticated:
        # Redirect to admin_panel if the user is already authenticated
        return redirect('admin_panel')
    
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            UserProfile.objects.all().delete()    
            Laser.objects.all().delete()
            Laser.objects.create(id='Red')
            Laser.objects.create(id='Green')
            Laser.objects.create(id='Purple')
            log(request, user)
            return redirect('admin_panel')
        else:
            error_message = "Username or password is incorrect."
            context = {
                'error_message': error_message,
                'gradient': True,
                'from_gradient': '#74EE15',
                'to_gradient': '#F000FF',
            }
            return render(request, 'login.html', context)
    
    
    context = {
        'gradient': True,
        'from_gradient': '#74EE15',
        'to_gradient': '#F000FF'
    }
    return render(request, 'login.html', context)

@login_required(login_url='login')
def logout(request):
    auth_logout(request)
    return redirect('login')

@login_required(login_url='login')
def admin_panel(request):
    IMAGE_DIR = str(_settings.BASE_DIR) + '/app/static/media'
    image_paths = None
    if os.path.exists(IMAGE_DIR):
        image_filenames = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.png', '.jpg', '.jpeg', '.gif', '.PNG'))]
    else:
        return JsonResponse({'error': f'Image directory does not exist, see: {IMAGE_DIR}'})
    
    context = {
        'gradient': True,
        'from_gradient': '#FFE700',
        'to_gradient': '#4DEEEA',
        'images': image_filenames,
        'video_feed': True,
        'users': UserProfile.objects.all(),
        'range': [0] * (3 - UserProfile.objects.count())
    } 
    return render(request, 'admin_panel.html', context)

