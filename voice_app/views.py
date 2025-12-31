from django.shortcuts import render
import os
import tempfile

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
from .ml_model import VoiceToTextModel

# Load model once
model = VoiceToTextModel()

def home(request):
    return render(request, 'voice_app/index.html')

@csrf_exempt
def convert_voice(request):
    """Handle voice recording"""
    if request.method == 'POST' and 'audio_data' in request.FILES:
        audio_file = request.FILES['audio_data']
        
        # Save temporarily
        fs = FileSystemStorage(location=tempfile.gettempdir())
        filename = fs.save(audio_file.name, audio_file)
        file_path = fs.path(filename)
        
        try:
            # Convert to text
            text = model.predict(file_path)
            
            # Clean up
            if os.path.exists(file_path):
                os.remove(file_path)
            
            return JsonResponse({
                'success': True,
                'text': text,
            })
        except Exception as e:
            # Clean up on error
            if os.path.exists(file_path):
                os.remove(file_path)
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)
    
    return JsonResponse({
        'success': False,
        'error': 'No audio received'
    }, status=400)

@csrf_exempt
def upload_voice(request):
    """Handle file upload"""
    if request.method == 'POST' and 'voice_file' in request.FILES:
        audio_file = request.FILES['voice_file']
        
        # Save temporarily
        fs = FileSystemStorage(location=tempfile.gettempdir())
        filename = fs.save(audio_file.name, audio_file)
        file_path = fs.path(filename)
        
        try:
            # Convert to text
            text = model.predict(file_path)
            
            # Clean up
            if os.path.exists(file_path):
                os.remove(file_path)
            
            return JsonResponse({
                'success': True,
                'text': text,
                'filename': audio_file.name
            })
        except Exception as e:
            # Clean up on error
            if os.path.exists(file_path):
                os.remove(file_path)
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)
    
    return JsonResponse({
        'success': False,
        'error': 'No file uploaded'
    }, status=400)