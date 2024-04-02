from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login as log, logout as auth_logout
from django.http import StreamingHttpResponse, JsonResponse
from django.views.decorators import gzip
from django.conf import settings as _settings
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponseRedirect, HttpResponseNotFound
from app.models import UserProfile, Laser
from django.urls import reverse
from screeninfo import get_monitors
import random
import cv2
import qrcode
import base64
import os
from io import BytesIO
from . import settings
import json
import requests
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

HOST = "http://localhost:8000"

def get_laser(request, laser_id):
    if request.method == 'GET':
        try:
            laser = Laser.objects.get(id=laser_id)
        except Laser.DoesNotExist:
            return errors(request, error_code=302)

        print(laser)
        return JsonResponse({
            'color': laser.color,
            'size': laser.size,
            'style': laser.style
        })
        
    return JsonResponse({'success': False, 'error': 'Invalid request method'}, status=405)

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

@login_required(login_url='login')
def remove_user_and_release_laser(request, first_name, last_name):
    user_to_del = get_object_or_404(UserProfile, first_name=first_name, last_name=last_name)
    
    if user_to_del:
        user_to_del.delete()
    return redirect(admin_panel)

#UC01, FR4
@login_required(login_url='login')
def signup(request):
    if request.method == 'POST':
        first_name = request.POST.get('firstname')
        last_name = request.POST.get('lastname')
        laser_pointer = request.POST.get('laser')
        laser = Laser.objects.get(id=laser_pointer)
        code = request.session.get('code', None)
        UserProfile.objects.create(first_name=first_name, last_name=last_name, laser=laser)
        
        if code:
            response = requests.get(f'{HOST}/api/v1/fetch_settings_url/{code}?firstname={first_name}&lastname={last_name}&laser={laser_pointer}') 
            redirect_url = None   
            if response.status_code == 200:
                response_data = response.json()
                redirect_url = response_data.get('url')

                if redirect_url:
                    print("Settings URL:", redirect_url)
                else:
                    print("Error: Failed to get the settings URL")
            else:
                print("Error:", response.status_code)

            if redirect_url:
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
                }
                return render(request, 'signup.html', context)
        else:
            laser = Laser.objects.get(id=laser_pointer)
            UserProfile.objects.create(first_name=first_name, last_name=last_name, laser=laser)
            base64_user_identifier = base64.b64encode(f"{first_name}_{last_name}_{laser_pointer}".encode('utf-8')).decode('utf-8')
            redirect_url = f"{HOST}/settings/{base64_user_identifier}"
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
            }
            return render(request, 'signup.html', context)
        
    lasers_without_users = Laser.objects.filter(userprofile__isnull=True)
    context = {
        'available_lasers': list(lasers_without_users.values_list('id', flat=True))
    }
        
    return render(request, 'signup.html', context)


def login(request):
    if request.user.is_authenticated:
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
            }
            return render(request, 'login.html', context)
    
    
    context = {    }
    return render(request, 'login.html', context)

@login_required(login_url='login')
def logout(request):
    auth_logout(request)
    return redirect('login')

@csrf_exempt
@login_required(login_url='login')
def store_code(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        code = data.get('code')
        request.session['code'] = code
        request.session.modified = True
    return redirect('admin_panel')

@login_required(login_url='login')
def del_code(request):
    if request.method == 'GET':
        del request.session['code']
        request.session.modified = True
    return redirect('admin_panel')

@login_required(login_url='login')
def admin_panel(request):
    code = request.session.get('code', None)
    connected = False
    if code:
        test_host = 'http://localhost:8000'
        prod_host = 'https://virtual-graffiti-box.onrender.com'

        code_validation_url = f"{test_host if settings.DEBUG else prod_host}/api/v1/validate_code/{code}"
        try:
            response = requests.get(code_validation_url)
            if response.ok:
                connected = True
        except requests.RequestException as e:
            print('Error:', e)
    try:
        absolute_path = os.path.abspath('virtual_graffiti/temp/reset_signal.txt')
        with open(absolute_path, 'r+') as f:
            reset_signal = int(f.read().strip())
            if reset_signal:
                f.seek(0)
                f.write('0')
                request.session['init'] = False
    except Exception as e:
        print(e)
        
    context = {
        'init': request.session.get('init', False),
        'video_feed': True,
        'users': UserProfile.objects.all(),
        'range': [0] * (3 - UserProfile.objects.count()), #NFR4, FR4
        'latency': 50,
        'cpu_usage': 80,
        'mem_usage': 60,
        'video_frames': 60,
        'connected': connected
    } 
    
    IMAGE_DIR = str(_settings.BASE_DIR) + '/app/static/media'
    if os.path.exists(IMAGE_DIR):
        image_filenames = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.png', '.jpg', '.jpeg', '.gif', '.PNG'))]
        context['images'] = image_filenames
    else:
        context['error'] = f'Image directory does not exist, see: {IMAGE_DIR}'    
    return render(request, 'admin_panel.html', context)


def errors(request, error_code=404):
    context = {
        'error_code': error_code
    }
    response = render(request, 'errors.html', context)
    response.status_code = error_code
    response['Location'] = '/errors/' + str(error_code)
    return response

@csrf_exempt
def check_reset_signal(request):
    try:
        absolute_path = os.path.abspath('virtual_graffiti/temp/reset_signal.txt')
        with open(absolute_path, 'r') as f:
            reset_signal = int(f.read().strip())
        return JsonResponse({'reset_signal': reset_signal})
    except Exception as e:
        print(e)
        return JsonResponse({'error': 'Error checking reset signal'}, status=500)