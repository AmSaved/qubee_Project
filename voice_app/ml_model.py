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
import warnings
warnings.filterwarnings('ignore')

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
            raise FileNotFoundError(f"Model not found at {model_path}")
        
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
            raise e
    
    def load_audio_any_format(self, audio_path):
        """Load audio from ANY format including AAC"""
        try:
            print(f"🔊 Loading audio: {audio_path}")
            
            # Method 1: Try librosa with all backends
            try:
                audio, sr = librosa.load(audio_path, sr=16000, mono=True)
                print(f"✅ Loaded with librosa, SR: {sr}, Length: {len(audio)/sr:.2f}s")
                return audio, sr
            except Exception as e:
                print(f"⚠️ Librosa failed: {e}")
            
            # Method 2: Try soundfile
            try:
                audio, sr = sf.read(audio_path)
                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)
                if sr != 16000:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                    sr = 16000
                print(f"✅ Loaded with soundfile, SR: {sr}, Length: {len(audio)/sr:.2f}s")
                return audio, sr
            except Exception as e:
                print(f"⚠️ Soundfile failed: {e}")
            
            # Method 3: Try ffmpeg/pydub (install: pip install pydub)
            try:
                from pydub import AudioSegment
                
                # Determine file type
                file_ext = os.path.splitext(audio_path)[1].lower()
                
                if file_ext == '.aac':
                    audio_seg = AudioSegment.from_file(audio_path, format="aac")
                elif file_ext == '.mp3':
                    audio_seg = AudioSegment.from_file(audio_path, format="mp3")
                elif file_ext == '.m4a':
                    audio_seg = AudioSegment.from_file(audio_path, format="m4a")
                else:
                    audio_seg = AudioSegment.from_file(audio_path)
                
                # Ensure mono before converting to numpy
                if audio_seg.channels > 1:
                    audio_seg = audio_seg.set_channels(1)
                
                # Convert to numpy
                samples = np.array(audio_seg.get_array_of_samples())
                sr = audio_seg.frame_rate
                
                # Convert to float32 and normalize
                if audio_seg.sample_width == 2:
                    samples = samples.astype(np.float32) / 32768.0
                elif audio_seg.sample_width == 1:
                    samples = samples.astype(np.float32) / 128.0
                    samples = samples - 1.0
                
                # Resample if needed
                if sr != 16000:
                    samples = librosa.resample(samples, orig_sr=sr, target_sr=16000)
                    sr = 16000
                
                print(f"✅ Loaded with pydub, SR: {sr}, Length: {len(samples)/sr:.2f}s")
                return samples, sr
                
            except Exception as e:
                print(f"⚠️ Pydub failed: {e}")
            
            # Method 4: Try raw binary reading for WAV files
            try:
                with open(audio_path, 'rb') as f:
                    data = f.read(4)
                
                # Check if it's a WAV file (starts with 'RIFF')
                if data == b'RIFF':
                    import wave
                    with wave.open(audio_path, 'rb') as wav:
                        sr = wav.getframerate()
                        frames = wav.readframes(wav.getnframes())
                        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                        
                        if wav.getnchannels() == 2:
                            audio = audio.reshape(-1, 2).mean(axis=1)
                        
                        if sr != 16000:
                            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                            sr = 16000
                        
                        print(f"✅ Loaded raw WAV, SR: {sr}, Length: {len(audio)/sr:.2f}s")
                        return audio, sr
            except:
                pass
            
            raise Exception(f"Could not load audio file: {audio_path}")
            
        except Exception as e:
            print(f"❌ All audio loading methods failed: {e}")
            raise
    
    def create_spectrogram(self, audio, sr):
        """Create spectrogram image"""
        try:
            # Ensure minimum length
            if len(audio) < sr:  # Less than 1 second
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
            print(f"🔤 Predicting letter from: {os.path.basename(audio_path)}")
            
            audio, sr = self.load_audio_any_format(audio_path)
            img = self.create_spectrogram(audio, sr)
            
            input_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(input_tensor)
            
            # Get probabilities
            probs = torch.nn.functional.softmax(output, dim=1)
            probs_np = probs.cpu().numpy()[0]
            
            # Get top 3 predictions
            top3_idx = np.argsort(probs_np)[-3:][::-1]
            top3_probs = probs_np[top3_idx]
            
            letters = "abcdefghijklmnopqrstuvwxyz"
            
            print("📊 Top 3 predictions:")
            for i, (idx, prob) in enumerate(zip(top3_idx, top3_probs)):
                if 0 <= idx < len(letters):
                    print(f"  {i+1}. '{letters[idx]}' ({prob*100:.1f}%)")
            
            # Return best prediction
            best_idx = top3_idx[0]
            if 0 <= best_idx < len(letters):
                letter = letters[best_idx]
                confidence = top3_probs[0] * 100
                print(f"✅ Best: '{letter}' ({confidence:.1f}% confidence)")
                return letter
            return "?"
            
        except Exception as e:
            print(f"❌ Letter prediction error: {e}")
            return "?"
    
    def predict(self, audio_path):
        """Main prediction - tries to predict word from full audio"""
        try:
            print(f"🎯 Predicting from: {os.path.basename(audio_path)}")
            return self.predict_letter(audio_path)
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return f"Error: {str(e)}"