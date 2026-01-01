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
        
        # Image transformations
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
            sys.exit(1)
        
        print(f"📦 Loading model from: {model_path}")
        
        try:
            # Create ResNet-18
            model = models.resnet18(weights='IMAGENET1K_V1')
            
            # Modify for 26 letters
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 26)
            
            # Load your weights
            state_dict = torch.load(str(model_path), map_location=self.device)
            model.load_state_dict(state_dict)
            
            model = model.to(self.device)
            model.eval()
            
            print(f"✅ ResNet-18 loaded with fine-tuned weights")
            return model
            
        except Exception as e:
            print(f"❌ ERROR loading model: {e}")
            sys.exit(1)
    
    def load_audio(self, audio_path):
        """Load audio file"""
        try:
            # Try librosa
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            return audio, sr
        except:
            # Try soundfile
            try:
                audio, sr = sf.read(audio_path)
                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)
                if sr != 16000:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                    sr = 16000
                return audio, sr
            except Exception as e:
                print(f"❌ Audio load error: {e}")
                raise
    
    def create_spectrogram(self, audio, sr):
        """Create spectrogram image"""
        try:
            # Ensure minimum length
            if len(audio) < sr:
                padding = sr - len(audio)
                audio = np.pad(audio, (0, padding), mode='constant')
            
            # Trim silence
            audio, _ = librosa.effects.trim(audio, top_db=25)
            
            # Mel spectrogram
            mel = librosa.feature.melspectrogram(
                y=audio, sr=sr, n_mels=128, n_fft=512, 
                hop_length=256, fmin=50, fmax=8000
            )
            
            # Log scale
            log_mel = librosa.power_to_db(mel, ref=np.max)
            
            # Normalize
            log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-10)
            log_mel = (log_mel * 255).astype(np.uint8)
            
            # Create image
            img = Image.fromarray(log_mel).convert('RGB')
            img = img.resize((224, 224), Image.Resampling.LANCZOS)
            
            return img
            
        except Exception as e:
            print(f"❌ Spectrogram error: {e}")
            raise
    
    def predict_letter(self, audio_path):
        """Predict single letter"""
        try:
            audio, sr = self.load_audio(audio_path)
            img = self.create_spectrogram(audio, sr)
            
            input_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(input_tensor)
            
            probs = torch.nn.functional.softmax(output, dim=1)
            _, pred_idx = torch.max(probs, 1)
            
            letters = "abcdefghijklmnopqrstuvwxyz"
            letter_idx = pred_idx.item()
            
            if 0 <= letter_idx < len(letters):
                return letters[letter_idx]
            return "?"
            
        except Exception as e:
            print(f"❌ Letter prediction error: {e}")
            return "?"
    
    def predict_word_simple(self, audio_path):
        """Simple word prediction using fixed segments"""
        try:
            audio, sr = self.load_audio(audio_path)
            
            # Fixed segment approach: 0.3 seconds per letter
            segment_duration = 0.3  # seconds
            segment_samples = int(segment_duration * sr)
            
            # Don't segment if audio is short
            if len(audio) <= segment_samples * 2:
                return self.predict_letter(audio_path)
            
            # Calculate how many segments
            max_segments = min(10, len(audio) // segment_samples)  # Max 10 letters
            
            if max_segments < 2:
                return self.predict_letter(audio_path)
            
            predicted_letters = []
            
            for i in range(max_segments):
                start = i * segment_samples
                end = start + segment_samples
                segment = audio[start:end]
                
                # Save segment
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    temp_path = f.name
                    sf.write(temp_path, segment, sr)
                
                try:
                    letter = self.predict_letter(temp_path)
                    if letter != "?":
                        predicted_letters.append(letter)
                finally:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
            
            if not predicted_letters:
                return self.predict_letter(audio_path)
            
            word = ''.join(predicted_letters)
            print(f"✅ Word: '{word}'")
            return word
            
        except Exception as e:
            print(f"❌ Word prediction error: {e}")
            return self.predict_letter(audio_path)
    
    def predict(self, audio_path):
        """Main prediction - tries to predict word"""
        return self.predict_word_simple(audio_path)
    
    def predict_sequence(self, audio_path, segment_duration=0.3):
        """Alias for predict_word_simple"""
        return self.predict_word_simple(audio_path)