import os
import tempfile
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

# Import model
try:
    from .ml_model import VoiceToTextModel
    model = VoiceToTextModel()
    MODEL_LOADED = True
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    MODEL_LOADED = False
    model = None

def home(request):
    return render(request, 'index.html')

@csrf_exempt
def convert_voice(request):
    if not MODEL_LOADED:
        return JsonResponse({
            'success': False,
            'error': 'Model not loaded. Check server console.'
        }, status=500)
    
    if request.method == 'POST' and 'audio_data' in request.FILES:
        audio_file = request.FILES['audio_data']
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f:
            for chunk in audio_file.chunks():
                f.write(chunk)
            temp_path = f.name
        
        try:
            # Get prediction
            text = model.predict(temp_path)
            
            # Cleanup
            os.unlink(temp_path)
            
            return JsonResponse({
                'success': True,
                'text': text
            })
            
        except Exception as e:
            # Cleanup on error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
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
    if not MODEL_LOADED:
        return JsonResponse({
            'success': False,
            'error': 'Model not loaded. Check server console.'
        }, status=500)
    
    if request.method == 'POST' and 'voice_file' in request.FILES:
        audio_file = request.FILES['voice_file']
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f:
            for chunk in audio_file.chunks():
                f.write(chunk)
            temp_path = f.name
        
        try:
            # Get prediction
            text = model.predict(temp_path)
            
            # Cleanup
            os.unlink(temp_path)
            
            return JsonResponse({
                'success': True,
                'text': text,
                'filename': audio_file.name
            })
            
        except Exception as e:
            # Cleanup on error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)
    
    return JsonResponse({
        'success': False,
        'error': 'No file uploaded'
    }, status=400)