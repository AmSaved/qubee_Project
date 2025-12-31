from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('convert/', views.convert_voice, name='convert_voice'),
    path('upload/', views.upload_voice, name='upload_voice'),
    path('status/', views.model_status, name='model_status'),
    path('test/', views.test_connection, name='test_connection'),
]