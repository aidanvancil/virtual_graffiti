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
    re_path(r'^.*/$', views.errors, name='errors'),
]