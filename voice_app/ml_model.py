import torch
import torch.nn as nn
import torchvision.models as models
import librosa
import numpy as np
import os
import sys
from pathlib import Path
import soundfile as sf
from PIL import Image

class VoiceToTextModel:
    def __init__(self):
        print("=== Loading Qubee Voice Model ===")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")
        
        # Load model
        self.model = self.load_model()
        print("✅ Model loaded successfully")
        
        # Image transformations (same as training)
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def load_model(self):
        """Load your trained ResNet-18 model"""
        model_path = Path("media/models/qubee_voice_model.pth")
        
        if not model_path.exists():
            print(f"❌ ERROR: Model not found at {model_path}")
            print(f"Place your trained model file at: {model_path.absolute()}")
            sys.exit(1)
        
        print(f"📦 Loading model from: {model_path}")
        
        try:
            # Create ResNet-18 model
            model = models.resnet18(pretrained=True)
            
            # Modify last layer for 26 letters
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 26)  # 26 letters
            
            # Load weights
            state_dict = torch.load(str(model_path), map_location=self.device)
            model.load_state_dict(state_dict)
            
            # Move to device and set eval
            model = model.to(self.device)
            model.eval()
            
            print(f"✅ ResNet-18 model loaded with 26 output classes")
            return model
            
        except Exception as e:
            print(f"❌ ERROR loading model: {e}")
            sys.exit(1)
    
    def load_audio_file(self, audio_path):
        """Load audio file with support for multiple formats including AAC"""
        try:
            # Try librosa first (supports many formats)
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            return audio, sr
        except:
            try:
                # Try soundfile for AAC and other formats
                audio, sr = sf.read(audio_path)
                # Convert to mono if stereo
                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)
                # Resample to 16000 if needed
                if sr != 16000:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                    sr = 16000
                return audio, sr
            except Exception as e:
                print(f"❌ Error loading audio file {audio_path}: {e}")
                raise
    
    def audio_to_spectrogram_image(self, audio, sr):
        """Convert audio to spectrogram image"""
        try:
            # Trim silence
            audio, _ = librosa.effects.trim(audio, top_db=20)
            
            # Ensure minimum length (1 second)
            if len(audio) < sr:
                padding = sr - len(audio)
                audio = np.pad(audio, (0, padding), mode='constant')
            
            # Create mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_mels=80,
                n_fft=512,
                hop_length=256,
                fmin=20,
                fmax=8000
            )
            
            # Convert to log scale
            log_mel = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize to 0-255 for image
            log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-10)
            log_mel = (log_mel * 255).astype(np.uint8)
            
            # Create PIL Image
            img = Image.fromarray(log_mel).convert('RGB')
            
            # Resize to 224x224
            img = img.resize((224, 224), Image.Resampling.LANCZOS)
            
            return img
            
        except Exception as e:
            print(f"❌ Error creating spectrogram: {e}")
            raise
    
    def predict(self, audio_path):
        """Convert audio to letter"""
        print(f"🎯 Processing single prediction: {Path(audio_path).name}")
        
        try:
            # Load audio
            audio, sr = self.load_audio_file(audio_path)
            
            # Convert to spectrogram image
            spectrogram_img = self.audio_to_spectrogram_image(audio, sr)
            
            # Apply transformations
            input_tensor = self.transform(spectrogram_img)
            input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
            input_tensor = input_tensor.to(self.device)
            
            # Model prediction
            with torch.no_grad():
                output = self.model(input_tensor)
            
            # Get predicted letter
            probabilities = torch.nn.functional.softmax(output, dim=1)
            _, predicted_idx = torch.max(probabilities, 1)
            predicted_class = predicted_idx.item()
            
            # Map to Qubee letter
            qubee_letters = "abcdefghijklmnopqrstuvwxyz"
            
            if 0 <= predicted_class < len(qubee_letters):
                predicted_letter = qubee_letters[predicted_class]
                print(f"✅ Predicted letter: '{predicted_letter}'")
                return predicted_letter
            else:
                return "?"
            
        except Exception as e:
            error_msg = f"Prediction error: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg
    
    def predict_sequence(self, audio_path, segment_duration=0.5):
        """
        Split audio into segments and predict each letter
        """
        print(f"🎯 Processing sequence: {Path(audio_path).name}")
        
        try:
            # Load full audio
            audio, sr = self.load_audio_file(audio_path)
            
            # Calculate segment size in samples
            segment_samples = int(segment_duration * sr)
            
            # Don't split if audio is too short
            if len(audio) <= segment_samples:
                print("⚠️ Audio too short for segmentation, using single prediction")
                return self.predict(audio_path)
            
            # How many segments we can get
            total_segments = len(audio) // segment_samples
            
            print(f"📊 Splitting into {total_segments} segments ({segment_duration}s each)")
            
            predicted_letters = []
            
            for i in range(total_segments):
                start_sample = i * segment_samples
                end_sample = start_sample + segment_samples
                segment = audio[start_sample:end_sample]
                
                # Save segment to temp file
                import tempfile
                temp_path = None
                try:
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                        temp_path = f.name
                        sf.write(temp_path, segment, sr)
                    
                    # Predict letter for this segment
                    letter = self.predict(temp_path)
                    predicted_letters.append(letter)
                    print(f"  Segment {i+1}/{total_segments}: '{letter}'")
                    
                finally:
                    # Cleanup temp file
                    if temp_path and os.path.exists(temp_path):
                        os.unlink(temp_path)
                
                # Break early if we have enough letters
                if len(predicted_letters) >= 10:  # Max 10 letters
                    break
            
            # Combine letters into word
            word = ''.join(predicted_letters)
            print(f"✅ Predicted sequence: '{word}'")
            return word
            
        except Exception as e:
            print(f"❌ Sequence prediction error: {e}")
            # Fallback to single prediction
            try:
                return self.predict(audio_path)
            except:
                return f"Error: {str(e)[:100]}"