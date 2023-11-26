from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login as log, logout as auth_logout
from django.http import StreamingHttpResponse
from django.views.decorators import gzip
from app.models import UserProfile, Laser
import cv2
import qrcode
import base64
from io import BytesIO

HOST = "localhost:8000"

@gzip.gzip_page
def video_feed(request):
    # Open the camera (adjust the camera index as needed, e.g., 0 for the default camera)
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
    
    laser_pointer = Laser.get(id=laser_pointer)
    user = UserProfile.objects.filter(first_name=first_name, last_name=last_name, laser=laser_pointer)
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
    context = {
        'gradient': True,
        'from_gradient': '#FFE700',
        'to_gradient': '#4DEEEA',
        'video_feed': True,
        'users': UserProfile.objects.all(),
        'range': [0] * (3 - UserProfile.objects.count())
    } 
    return render(request, 'admin_panel.html', context)

