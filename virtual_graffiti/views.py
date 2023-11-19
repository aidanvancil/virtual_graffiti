from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login as log, logout as auth_logout
import qrcode
import base64
from io import BytesIO

HOST = "localhost:8000"


def errors(request):
    context = {'error': 404}
    return render(request, 'errors.html', context)

def settings(request, user_identifier):
    user_identifier_decoded = base64.b64decode(user_identifier).decode('utf-8')
    first_name, laser_pointer = user_identifier_decoded.split('_')

    context = {
        'gradient': True,
        'from_gradient': '#74EE15',
        'to_gradient': '#F000FF',
        'first_name': first_name,
        'laser_pointer': laser_pointer,
    }

    return render(request, 'settings.html', context)

def signup(request):
    if request.method == 'POST':
        first_name = request.POST.get('firstname')
        laser_pointer = request.POST.get('laser')

        base64_user_identifier = base64.b64encode(f"{first_name}_{laser_pointer}".encode('utf-8')).decode('utf-8')
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
        }

        return render(request, 'signup.html', context)

    context = {
        'gradient': True,
        'from_gradient': '#74EE15',
        'to_gradient': '#F000FF',
    }
    return render(request, 'signup.html', context)


def login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None:    
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
    context = {}
    return render(request, 'homepage.html', context)

