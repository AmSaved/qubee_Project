import os
import tempfile
import json
import time
import subprocess
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import traceback

# Import model
try:
    from .ml_model import VoiceToTextModel
    model = VoiceToTextModel()
    MODEL_LOADED = True
    print("✅ Voice model initialized successfully")
except Exception as e:
    print(f"❌ Failed to load voice model: {e}")
    traceback.print_exc()
    MODEL_LOADED = False
    model = None

def ensure_wav_format(audio_path):
    """Convert any audio format to WAV"""
    try:
        # Check if already WAV
        if audio_path.lower().endswith('.wav'):
            return audio_path
        
        # Convert using ffmpeg if available
        wav_path = audio_path + '.wav'
        
        try:
            subprocess.run([
                'ffmpeg', '-i', audio_path, 
                '-acodec', 'pcm_s16le',
                '-ac', '1',
                '-ar', '16000',
                wav_path,
                '-y'
            ], check=True, capture_output=True, timeout=10)
            
            # Remove original
            if os.path.exists(audio_path):
                os.unlink(audio_path)
            return wav_path
            
        except Exception as e:
            print(f"⚠️ FFmpeg conversion failed: {e}")
            # If ffmpeg fails, try to use the original
            return audio_path
            
    except Exception as e:
        print(f"⚠️ Format conversion failed: {e}")
        return audio_path

def home(request):
    """Render the main page"""
    try:
        return render(request, 'index.html')
    except:
        # Simple fallback
        return HttpResponse("""
        <!DOCTYPE html>
        <html>
        <head><title>Qubee Voice</title></head>
        <body>
            <h1>🎤 Qubee Voice to Text</h1>
            <p>Model: """ + ("✅ LOADED" if MODEL_LOADED else "❌ NOT LOADED") + """</p>
        </body>
        </html>
        """)

@csrf_exempt
def convert_voice(request):
    """Handle voice recording from microphone"""
    if not MODEL_LOADED:
        return JsonResponse({
            'success': False,
            'error': 'Model not loaded',
            'text': ''
        }, status=500)
    
    if request.method == 'POST' and 'audio_data' in request.FILES:
        audio_file = request.FILES['audio_data']
        
        # Use original extension if possible, otherwise default to .wav
        file_ext = os.path.splitext(audio_file.name)[1]
        if not file_ext:
            file_ext = '.wav'
            
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as f:
            for chunk in audio_file.chunks():
                f.write(chunk)
            temp_path = f.name
        
        print(f"🎤 Received recording, saved to: {temp_path}")
        print(f"File size: {os.path.getsize(temp_path)} bytes")
        
        try:
            # Try to ensure it's WAV format
            temp_path = ensure_wav_format(temp_path)
            
            # Predict
            text = model.predict(temp_path)
            
            print(f"✅ Prediction: '{text}'")
            
            # Cleanup
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            
            return JsonResponse({
                'success': True,
                'text': text
            })
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            # Cleanup on error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            return JsonResponse({
                'success': False,
                'error': str(e),
                'text': ''
            }, status=500)
    
    return JsonResponse({
        'success': False,
        'error': 'No audio received'
    }, status=400)

@csrf_exempt
def upload_voice(request):
    """Handle uploaded audio file"""
    if not MODEL_LOADED:
        return JsonResponse({
            'success': False,
            'error': 'Model not loaded',
            'text': ''
        }, status=500)
    
    if request.method == 'POST' and 'voice_file' in request.FILES:
        audio_file = request.FILES['voice_file']
        
        print(f"📁 Uploading: {audio_file.name} ({audio_file.size} bytes)")
        
        # Size limit
        if audio_file.size > 50 * 1024 * 1024:
            return JsonResponse({
                'success': False,
                'error': 'File too large (max 50MB)',
                'text': ''
            }, status=400)
        
        # Save with original extension first
        file_ext = os.path.splitext(audio_file.name)[1]
        if not file_ext:
            file_ext = '.wav'
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as f:
            for chunk in audio_file.chunks():
                f.write(chunk)
            temp_path = f.name
        
        print(f"💾 Saved to: {temp_path}")
        
        try:
            # Convert to WAV if needed
            temp_path = ensure_wav_format(temp_path)
            
            # Predict
            text = model.predict(temp_path)
            
            print(f"✅ Prediction: '{text}'")
            
            # Cleanup
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            
            return JsonResponse({
                'success': True,
                'text': text,
                'filename': audio_file.name
            })
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            return JsonResponse({
                'success': False,
                'error': str(e),
                'text': ''
            }, status=500)
    
    return JsonResponse({
        'success': False,
        'error': 'No file uploaded'
    }, status=400)

def model_status(request):
    """Check model status"""
    return JsonResponse({
        'model_loaded': MODEL_LOADED,
        'status': 'ready' if MODEL_LOADED else 'error',
        'message': 'ResNet-18 model ready' if MODEL_LOADED else 'Model failed'
    })

def test_connection(request):
    """Test API connection"""
    return JsonResponse({
        'success': True,
        'message': 'Qubee Voice API running',
        'model_loaded': MODEL_LOADED
    })