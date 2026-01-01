import os
import tempfile
import json
import time
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

# Import model
try:
    from .ml_model import VoiceToTextModel
    model = VoiceToTextModel()
    MODEL_LOADED = True
    print("✅ Voice model initialized successfully")
except Exception as e:
    print(f"❌ Failed to load voice model: {e}")
    MODEL_LOADED = False
    model = None

def home(request):
    """Render the main page"""
    # Check if template exists first
    template_path = os.path.join(os.path.dirname(__file__), 'templates', 'index.html')
    
    if os.path.exists(template_path):
        try:
            return render(request, 'index.html')
        except Exception as e:
            print(f"Template render error: {e}")
            return HttpResponse(f"Template error: {e}")
    else:
        # Template doesn't exist, create simple page
        print(f"Template not found at: {template_path}")
        return HttpResponse("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Qubee Voice to Text</title>
            <style>
                body { font-family: Arial; padding: 20px; }
                .container { max-width: 600px; margin: 0 auto; }
                .card { border: 1px solid #ddd; padding: 20px; margin: 20px 0; border-radius: 10px; }
                button { padding: 10px 20px; margin: 5px; cursor: pointer; }
                #recordBtn { background: green; color: white; }
                #stopBtn { background: red; color: white; }
                #result { padding: 15px; background: #f5f5f5; min-height: 100px; margin: 10px 0; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🎤 Qubee Afan Oromo Voice to Text</h1>
                <p>Model status: """ + ("✅ LOADED" if MODEL_LOADED else "❌ NOT LOADED") + """</p>
                
                <div class="card">
                    <h2>Record Voice</h2>
                    <button id="recordBtn">🎤 Start Recording</button>
                    <button id="stopBtn" disabled>⏹ Stop</button>
                    <div id="recordStatus">Ready</div>
                </div>
                
                <div class="card">
                    <h2>Upload File</h2>
                    <input type="file" id="audioFile" accept="audio/*">
                    <button id="uploadBtn">📁 Convert File</button>
                    <div id="fileStatus">No file selected</div>
                </div>
                
                <div class="card">
                    <h2>Converted Text</h2>
                    <div id="result">Text will appear here...</div>
                    <button id="copyBtn" style="display:none;">📋 Copy</button>
                </div>
            </div>
            
            <script>
                // Basic JavaScript
                document.getElementById('recordBtn').onclick = () => {
                    alert("Recording would start here");
                };
                
                document.getElementById('uploadBtn').onclick = () => {
                    alert("File upload would start here");
                };
                
                // Test API endpoints
                fetch('/status/').then(r => r.json()).then(data => {
                    console.log("Model status:", data);
                });
            </script>
        </body>
        </html>
        """)

@csrf_exempt
def convert_voice(request):
    """Handle voice recording from microphone"""
    if not MODEL_LOADED:
        return JsonResponse({
            'success': False,
            'error': 'Voice model not loaded. Please check server logs.',
            'text': '',
            'timestamp': time.time()
        }, status=500)
    
    start_time = time.time()
    
    # Check for audio file
    if request.method == 'POST' and 'audio_data' in request.FILES:
        audio_file = request.FILES['audio_data']
        
        # File information
        file_info = {
            'name': audio_file.name,
            'size': audio_file.size,
            'type': audio_file.content_type,
            'received_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        print(f"🎤 Received voice recording: {file_info}")
        
        # Save to temporary file
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.aac') as f:
                for chunk in audio_file.chunks():
                    f.write(chunk)
                temp_path = f.name
            
            print(f"💾 Saved to temporary file: {temp_path}")
            
            # Process with model
            print("🔄 Processing audio with ResNet-18 model...")
            
            # Try sequence prediction first
            try:
                text = model.predict_sequence(temp_path, segment_duration=0.5)
                if not text or text == "?" or len(text) < 1:
                    text = model.predict(temp_path)
            except:
                text = model.predict(temp_path)
            
            processing_time = time.time() - start_time
            
            print(f"✅ Prediction completed in {processing_time:.2f} seconds")
            print(f"📝 Result: '{text}'")
            
            # Clean up temporary file
            try:
                os.unlink(temp_path)
                print(f"🗑️ Cleaned up temporary file")
            except:
                pass
            
            return JsonResponse({
                'success': True,
                'text': text,
                'type': 'voice_recording',
                'processing_time': round(processing_time, 2),
                'model_type': 'ResNet-18',
                'timestamp': time.time()
            })
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Error processing audio: {str(e)}"
            print(f"❌ {error_msg}")
            
            # Clean up on error
            if 'temp_path' in locals() and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                    print(f"🗑️ Cleaned up temporary file after error")
                except:
                    pass
            
            return JsonResponse({
                'success': False,
                'error': error_msg,
                'text': '',
                'processing_time': round(processing_time, 2),
                'timestamp': time.time()
            }, status=500)
    
    return JsonResponse({
        'success': False,
        'error': 'No audio data received.',
        'text': '',
        'timestamp': time.time()
    }, status=400)

@csrf_exempt
def upload_voice(request):
    """Handle uploaded audio file"""
    if not MODEL_LOADED:
        return JsonResponse({
            'success': False,
            'error': 'Voice model not loaded. Please check server logs.',
            'text': '',
            'timestamp': time.time()
        }, status=500)
    
    start_time = time.time()
    
    if request.method == 'POST' and 'voice_file' in request.FILES:
        audio_file = request.FILES['voice_file']
        
        # File validation
        file_info = {
            'name': audio_file.name,
            'size': audio_file.size,
            'type': audio_file.content_type,
            'uploaded_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        print(f"📁 Received file upload: {file_info}")
        
        # Size limit: 50MB
        MAX_SIZE = 50 * 1024 * 1024  # 50MB
        if audio_file.size > MAX_SIZE:
            return JsonResponse({
                'success': False,
                'error': f'File too large. Maximum size is {MAX_SIZE/(1024*1024):.0f}MB.',
                'text': '',
                'timestamp': time.time()
            }, status=400)
        
        # Allowed file extensions
        allowed_extensions = ['.wav', '.mp3', '.m4a', '.ogg', '.flac', '.webm', '.aac']
        filename = audio_file.name.lower()
        
        if not any(filename.endswith(ext) for ext in allowed_extensions):
            return JsonResponse({
                'success': False,
                'error': f'Unsupported file type. Allowed: {", ".join([ext for ext in allowed_extensions])}',
                'text': '',
                'timestamp': time.time()
            }, status=400)
        
        # Save to temporary file
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f:
                for chunk in audio_file.chunks():
                    f.write(chunk)
                temp_path = f.name
            
            print(f"💾 Saved to temporary file: {temp_path}")
            
            # Process with model
            print(f"🔄 Processing uploaded file with ResNet-18 model...")
            
            # Try sequence prediction first
            try:
                text = model.predict_sequence(temp_path, segment_duration=0.5)
                if not text or text == "?" or len(text) < 1:
                    text = model.predict(temp_path)
            except:
                text = model.predict(temp_path)
            
            processing_time = time.time() - start_time
            
            print(f"✅ File processing completed in {processing_time:.2f} seconds")
            print(f"📝 Result: '{text}'")
            
            # Clean up temporary file
            try:
                os.unlink(temp_path)
                print(f"🗑️ Cleaned up temporary file")
            except:
                pass
            
            return JsonResponse({
                'success': True,
                'text': text,
                'type': 'file_upload',
                'filename': audio_file.name,
                'file_size': audio_file.size,
                'processing_time': round(processing_time, 2),
                'model_type': 'ResNet-18',
                'timestamp': time.time()
            })
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Error processing file: {str(e)}"
            print(f"❌ {error_msg}")
            
            # Clean up on error
            if 'temp_path' in locals() and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                    print(f"🗑️ Cleaned up temporary file after error")
                except:
                    pass
            
            return JsonResponse({
                'success': False,
                'error': error_msg,
                'text': '',
                'processing_time': round(processing_time, 2),
                'timestamp': time.time()
            }, status=500)
    
    return JsonResponse({
        'success': False,
        'error': 'No file uploaded.',
        'text': '',
        'timestamp': time.time()
    }, status=400)

def model_status(request):
    """Check model status and health"""
    status_info = {
        'model_loaded': MODEL_LOADED,
        'status': 'ready' if MODEL_LOADED else 'error',
        'message': 'ResNet-18 model is loaded and ready' if MODEL_LOADED else 'Model failed to load',
        'model_type': 'ResNet-18 (26-class classification)',
        'timestamp': time.time(),
        'endpoints': {
            'home': '/',
            'record_voice': '/convert/',
            'upload_file': '/upload/',
            'status': '/status/',
            'test': '/test/'
        }
    }
    
    # Add system info if model is loaded
    if MODEL_LOADED and hasattr(model, 'device'):
        status_info['device'] = str(model.device)
    
    return JsonResponse(status_info)

def test_connection(request):
    """Test API connection and basic functionality"""
    test_data = {
        'success': True,
        'message': 'Qubee Afan Oromo Voice API is running',
        'service': 'Voice-to-Text Conversion',
        'language': 'Afan Oromo (Qubee script)',
        'model_loaded': MODEL_LOADED,
        'model_type': 'ResNet-18 image classification',
        'timestamp': time.time(),
        'version': '1.0.0',
        'features': [
            'Real-time voice recording',
            'Audio file upload',
            'Afan Oromo speech recognition',
            'Qubee text output'
        ]
    }
    
    return JsonResponse(test_data)

def health_check(request):
    """Health check endpoint for monitoring"""
    health_status = {
        'status': 'healthy' if MODEL_LOADED else 'unhealthy',
        'model': 'loaded' if MODEL_LOADED else 'failed',
        'timestamp': time.time(),
        'uptime': time.time() - start_time if 'start_time' in globals() else 0
    }
    
    return JsonResponse(health_status)

# Record startup time for uptime calculation
start_time = time.time()