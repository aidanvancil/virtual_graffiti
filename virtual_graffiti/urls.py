"""
URL configuration for virtual_graffiti project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include, re_path
from . import views

'''   
    Author(s): Foster Schmidt (F), Moises Moreno (M), Aidan Vancil (A)
    Date(s):   11/12/23 - 12/03/23
    
    Description:
    - (A + F) admin, admin_panel, settings
    - (M + F) register, login
    - (A + M) models
    - (M) errors
    - (A) set_laser_*, get_laser_*, disconnect
    - (F) logout
'''

urlpatterns = [
    path("__reload__/", include("django_browser_reload.urls")),
    path('admin/', admin.site.urls),
    path('admin_panel/', views.admin_panel, name='admin_panel'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('', views.login, name='login'),
    path('register/', views.signup, name='signup'),
    path('logout/', views.logout, name='logout'),
    path('settings/<str:user_identifier>/', views.settings, name='settings'),
    path('disconnect/<str:first_name>_<str:last_name>/', views.remove_user_and_release_laser, name='disconnect_user'),
    path('initialize_projector/', views.initialize_projector, name='initialize_projector'),
    path('set_laser_color/<str:laser_id>/', views.set_laser_color, name='set_laser_color'),
    path('set_laser_size/<str:laser_id>/', views.set_laser_size, name='set_laser_size'),
    path('set_laser_style/<str:laser_id>/', views.set_laser_style, name='set_laser_style'),
    path('get_laser/<str:laser_id>/', views.get_laser, name='get_laser'),
    re_path(r'^.*/$', views.errors, name='errors'),
]