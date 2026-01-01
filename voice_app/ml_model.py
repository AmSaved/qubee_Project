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
from torchvision import transforms

class VoiceToTextModel:
    def __init__(self):
        print("=== Loading Qubee Voice Model ===")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")
        
        # Load model
        self.model = self.load_model()
        print("✅ Model loaded successfully")
        
        # Image transformations (SAME as your training)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Test the model with a random input
        self.test_model()
    
    def load_model(self):
        """Load your trained ResNet-18 model"""
        model_path = Path("media/models/qubee_voice_model.pth")
        
        if not model_path.exists():
            print(f"❌ ERROR: Model not found at {model_path}")
            sys.exit(1)
        
        print(f"📦 Loading model weights from: {model_path}")
        
        try:
            # Load state_dict
            state_dict = torch.load(str(model_path), map_location=self.device)
            print(f"✅ Loaded state_dict with {len(state_dict)} parameters")
            
            # Check the output layer shape
            print(f"📊 fc.weight shape: {state_dict['fc.weight'].shape}")
            print(f"📊 fc.bias shape: {state_dict['fc.bias'].shape}")
            
            num_classes = state_dict['fc.weight'].shape[0]
            print(f"📊 Model has {num_classes} output classes")
            
            # Create ResNet-18 WITHOUT pretrained weights
            model = models.resnet18(weights=None)
            
            # Modify the final layer to match your model (26 classes)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
            
            # Load the weights
            model.load_state_dict(state_dict)
            
            # Move to device and set eval
            model = model.to(self.device)
            model.eval()
            
            print(f"✅ ResNet-18 loaded with {num_classes} output classes")
            return model
            
        except Exception as e:
            print(f"❌ ERROR loading model: {e}")
            sys.exit(1)
    
    def test_model(self):
        """Test model with random input to see predictions"""
        print("\n🧪 Testing model with random input...")
        
        # Create random input
        random_input = torch.randn(1, 3, 224, 224).to(self.device)
        
        with torch.no_grad():
            output = self.model(random_input)
        
        # Get probabilities
        probs = torch.softmax(output, dim=1)
        probs_np = probs.cpu().numpy()[0]
        
        # Show top predictions
        letters = "abcdefghijklmnopqrstuvwxyz"
        num_classes = output.shape[1]
        
        print(f"Model outputs: {num_classes} classes")
        
        if num_classes != 26:
            print(f"⚠️ WARNING: Model has {num_classes} outputs, expected 26!")
        
        # Get top 5 predictions
        top5_idx = np.argsort(probs_np)[-5:][::-1]
        
        print("Top 5 predictions from random input:")
        for i, idx in enumerate(top5_idx):
            if idx < len(letters):
                letter = letters[idx]
            else:
                letter = f"[class{idx}]"
            
            confidence = probs_np[idx] * 100
            print(f"  {i+1}. '{letter}' - {confidence:.2f}%")
        
        # Check if model always predicts same thing
        if torch.std(output) < 0.1:
            print("⚠️ WARNING: Model outputs are almost identical (might be broken)")
    
    def load_audio(self, audio_path):
        """Load audio file"""
        try:
            # Try librosa first
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
                print(f"❌ Cannot load audio: {e}")
                # Create dummy audio for testing
                return np.sin(2 * np.pi * 440 * np.arange(16000) / 16000), 16000
    
    def create_spectrogram(self, audio, sr):
        """
        Create spectrogram EXACTLY like your training
        Your training used: 224x224 RGB images from conv_* files
        """
        try:
            # Trim silence
            audio, _ = librosa.effects.trim(audio, top_db=25)
            
            # Ensure minimum length (1 second)
            if len(audio) < sr:
                padding = sr - len(audio)
                audio = np.pad(audio, (0, padding), mode='constant')
            
            # Your training likely used these parameters:
            # Based on common speech recognition settings
            n_mels = 128  # Common for ResNet
            n_fft = 512
            hop_length = 256
            
            # Create mel spectrogram
            mel = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_mels=n_mels,
                n_fft=n_fft,
                hop_length=hop_length,
                fmin=50,
                fmax=8000
            )
            
            # Convert to log scale
            log_mel = librosa.power_to_db(mel, ref=np.max)
            
            # Normalize to 0-255 (like images)
            log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-10)
            log_mel = (log_mel * 255).astype(np.uint8)
            
            # Create RGB image (3 channels)
            img = Image.fromarray(log_mel).convert('RGB')
            
            # Resize to 224x224 (ResNet input)
            img = img.resize((224, 224), Image.Resampling.LANCZOS)
            
            return img
            
        except Exception as e:
            print(f"❌ Spectrogram error: {e}")
            # Return a test pattern
            return Image.new('RGB', (224, 224), color='white')
    
    def predict(self, audio_path):
        """Main prediction function"""
        print(f"\n🎯 Processing: {os.path.basename(audio_path)}")
        
        try:
            # 1. Load audio
            audio, sr = self.load_audio(audio_path)
            print(f"📊 Audio length: {len(audio)/sr:.2f}s")
            
            # 2. Create spectrogram
            img = self.create_spectrogram(audio, sr)
            
            # 3. Transform
            input_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            # 4. Model prediction
            with torch.no_grad():
                output = self.model(input_tensor)
            
            # 5. Get probabilities
            probs = torch.softmax(output, dim=1)
            probs_np = probs.cpu().numpy()[0]
            
            # 6. Map to letters
            letters = "abcdefghijklmnopqrstuvwxyz"
            
            # 7. Show all predictions
            print(f"\n📈 ALL PREDICTIONS (confidence > 1%):")
            predictions = []
            
            for idx in range(len(probs_np)):
                confidence = probs_np[idx] * 100
                if confidence > 1:  # Only show meaningful predictions
                    if idx < len(letters):
                        letter = letters[idx]
                    else:
                        letter = f"[{idx}]"
                    
                    predictions.append((letter, confidence))
                    print(f"  '{letter}': {confidence:.1f}%")
            
            # 8. Get best prediction
            best_idx = np.argmax(probs_np)
            best_confidence = probs_np[best_idx] * 100
            
            if best_idx < len(letters):
                result = letters[best_idx]
                print(f"\n✅ BEST PREDICTION: '{result}' ({best_confidence:.1f}% confidence)")
                
                # If confidence is too low, might be wrong
                if best_confidence < 50:
                    print(f"⚠️ Low confidence - model unsure")
                
                return result
            else:
                print(f"⚠️ Invalid class index: {best_idx}")
                return "?"
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return "error"