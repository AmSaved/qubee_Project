import os
import tempfile
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import json

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
    """Handle voice recording from microphone"""
    if not MODEL_LOADED:
        return JsonResponse({
            'success': False,
            'error': 'Model not loaded. Please check server console for errors.',
            'text': ''
        }, status=500)
    
    if request.method == 'POST' and 'audio_data' in request.FILES:
        audio_file = request.FILES['audio_data']
        
        # Get audio duration from metadata if available
        audio_info = {
            'name': audio_file.name,
            'size': audio_file.size,
            'content_type': audio_file.content_type
        }
        
        print(f"🎤 Received audio recording: {audio_info}")
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f:
            for chunk in audio_file.chunks():
                f.write(chunk)
            temp_path = f.name
        
        print(f"💾 Saved temp file: {temp_path}")
        
        try:
            # Get prediction
            print("🔄 Processing audio with model...")
            text = model.predict(temp_path)
            
            # If we only get a single character, try sequence decoding
            if len(text) == 1 and text.isalpha():
                print("🔄 Single character detected, trying sequence decoding...")
                sequence_text = model.decode_sequence(temp_path)
                if len(sequence_text) > 1:
                    text = sequence_text
                    print(f"✅ Sequence decoding gave: {text}")
            
            print(f"✅ Final prediction: '{text}'")
            
            # Cleanup temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                print(f"🗑️ Cleaned up temp file: {temp_path}")
            
            return JsonResponse({
                'success': True,
                'text': text,
                'type': 'recording',
                'info': audio_info
            })
            
        except Exception as e:
            print(f"❌ Error during conversion: {str(e)}")
            
            # Cleanup on error
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                    print(f"🗑️ Cleaned up temp file after error: {temp_path}")
                except:
                    pass
            
            return JsonResponse({
                'success': False,
                'error': f'Conversion failed: {str(e)}',
                'text': ''
            }, status=500)
    
    # Check if JSON data was sent (alternative method)
    elif request.method == 'POST' and request.body:
        try:
            data = json.loads(request.body)
            if 'audio_base64' in data:
                # Handle base64 audio (not implemented yet)
                return JsonResponse({
                    'success': False,
                    'error': 'Base64 audio not supported yet',
                    'text': ''
                }, status=400)
        except:
            pass
    
    return JsonResponse({
        'success': False,
        'error': 'No audio data received. Please record or upload audio.',
        'text': ''
    }, status=400)

@csrf_exempt
def upload_voice(request):
    """Handle uploaded audio file"""
    if not MODEL_LOADED:
        return JsonResponse({
            'success': False,
            'error': 'Model not loaded. Please check server console for errors.',
            'text': ''
        }, status=500)
    
    if request.method == 'POST' and 'voice_file' in request.FILES:
        audio_file = request.FILES['voice_file']
        
        # Validate file
        if audio_file.size > 50 * 1024 * 1024:  # 50MB limit
            return JsonResponse({
                'success': False,
                'error': 'File too large (max 50MB)',
                'text': ''
            }, status=400)
        
        # Check file extension
        allowed_extensions = ['.wav', '.mp3', '.m4a', '.ogg', '.flac', '.webm']
        filename = audio_file.name.lower()
        if not any(filename.endswith(ext) for ext in allowed_extensions):
            return JsonResponse({
                'success': False,
                'error': f'Unsupported file type. Allowed: {", ".join(allowed_extensions)}',
                'text': ''
            }, status=400)
        
        audio_info = {
            'name': audio_file.name,
            'size': audio_file.size,
            'content_type': audio_file.content_type,
            'type': 'upload'
        }
        
        print(f"📁 Received file upload: {audio_info}")
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f:
            for chunk in audio_file.chunks():
                f.write(chunk)
            temp_path = f.name
        
        print(f"💾 Saved temp file: {temp_path}")
        
        try:
            # Get prediction
            print("🔄 Processing uploaded file with model...")
            text = model.predict(temp_path)
            
            # If we only get a single character, try sequence decoding
            if len(text) == 1 and text.isalpha():
                print("🔄 Single character detected, trying sequence decoding...")
                sequence_text = model.decode_sequence(temp_path)
                if len(sequence_text) > 1:
                    text = sequence_text
                    print(f"✅ Sequence decoding gave: {text}")
            
            print(f"✅ Final prediction: '{text}'")
            
            # Cleanup temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                print(f"🗑️ Cleaned up temp file: {temp_path}")
            
            return JsonResponse({
                'success': True,
                'text': text,
                'filename': audio_file.name,
                'size': audio_file.size,
                'type': 'upload'
            })
            
        except Exception as e:
            print(f"❌ Error during file conversion: {str(e)}")
            
            # Cleanup on error
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                    print(f"🗑️ Cleaned up temp file after error: {temp_path}")
                except:
                    pass
            
            return JsonResponse({
                'success': False,
                'error': f'File conversion failed: {str(e)}',
                'text': ''
            }, status=500)
    
    return JsonResponse({
        'success': False,
        'error': 'No file uploaded. Please select an audio file.',
        'text': ''
    }, status=400)

def model_status(request):
    """Check model status"""
    return JsonResponse({
        'model_loaded': MODEL_LOADED,
        'status': 'ready' if MODEL_LOADED else 'error',
        'message': 'Model is loaded and ready' if MODEL_LOADED else 'Model failed to load'
    })

def test_connection(request):
    """Test API connection"""
    return JsonResponse({
        'success': True,
        'message': 'Qubee Voice API is running',
        'model_status': 'loaded' if MODEL_LOADED else 'not_loaded',
        'endpoints': {
            'home': '/',
            'record_voice': '/convert/',
            'upload_file': '/upload/',
            'status': '/status/',
            'test': '/test/'
        }
    })