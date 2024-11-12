"""project_settings URL Configuration
"""
from django.contrib import admin
from django.urls import path, include
from . import views
from .views import user_login, user_logout, register, about, index, predict_page, cuda_full

app_name = 'ml_app'
handler404 = views.handler404

urlpatterns = [
    path('home/', index, name='index'),
    path('', user_login ,name="login"),
    path('logout', user_logout,name="logout"),
    path('register/', register ,name="register"),
    path('about/', about, name='about'),
    path('predict/', predict_page, name='predict'),
    path('cuda_full/',cuda_full,name='cuda_full'),
]
