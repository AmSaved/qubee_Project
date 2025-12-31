from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('convert/', views.convert_voice, name='convert_voice'),
    path('upload/', views.upload_voice, name='upload_voice'),
]