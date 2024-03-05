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
from . import views

#UC06
def settings(request, user_identifier):
    try:
        user_identifier_decoded = base64.b64decode(user_identifier).decode('utf-8')
        first_name, last_name, laser_pointer_id = user_identifier_decoded.split('_')
    except:
        return views.errors(request, error_code=302)
    
    try:
        laser_pointer = Laser.objects.get(id=laser_pointer_id)
        user = UserProfile.objects.get(first_name=first_name, last_name=last_name, laser=laser_pointer)
    except UserProfile.DoesNotExist:
        print("User profile does not exist.")
        return views.errors(request, error_code=302)
    except Laser.DoesNotExist:
        print("Laser does not exist.")
        return views.errors(request, error_code=302)
    except Exception as e:
        print("Error retrieving user profile:", e)
        return views.errors(request, error_code=500)
    
    context = {
        'gradient': True,
        'from_gradient': '#74EE15',
        'to_gradient': '#F000FF',
        'first_name': user.first_name,
        'last_name': user.last_name,
        'laser_pointer': user.laser.id,
    }

    return render(request, 'settings.html', context)