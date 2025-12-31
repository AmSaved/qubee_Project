import torch
import torch.nn as nn
import torchvision.models as models
import librosa
import numpy as np
import os
import sys
from pathlib import Path
from PIL import Image
import io

# ============================================
# YOUR EXACT MODEL ARCHITECTURE
# ============================================

class VoiceToTextModel:
    def __init__(self):
        print("=== Loading Qubee Voice Model ===")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")
        
        # Load model
        self.model = self.load_model()
        print("✅ Model loaded successfully")
        
        # Image transformations (SAME as your training)
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
            # Create ResNet-18 model (EXACTLY like your training)
            model = models.resnet18(pretrained=False)
            
            # Modify last layer for 26 letters (Afan Oromo alphabet)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 26)  # 26 letters
            
            # Load weights
            state_dict = torch.load(str(model_path), map_location=self.device)
            model.load_state_dict(state_dict)
            
            # Move to device and set eval
            model = model.to(self.device)
            model.eval()
            
            print(f"✅ ResNet-18 model loaded with 26 output classes")
            print(f"📊 Model type: ResNet-18 (image classification)")
            
            return model
            
        except Exception as e:
            print(f"❌ ERROR loading model: {e}")
            print("\n🔧 Make sure your .pth file matches ResNet-18 architecture")
            sys.exit(1)
    
    def audio_to_spectrogram_image(self, audio_path):
        """Convert audio to spectrogram image (SAME as your training data)"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            
            # Trim silence
            audio, _ = librosa.effects.trim(audio, top_db=20)
            
            # Create mel spectrogram (80 bands, like your training)
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
            from PIL import Image
            img = Image.fromarray(log_mel).convert('RGB')
            
            # Resize to 224x224 (ResNet input)
            img = img.resize((224, 224), Image.Resampling.LANCZOS)
            
            return img
            
        except Exception as e:
            print(f"❌ Error creating spectrogram: {e}")
            raise
    
    def predict(self, audio_path):
        """Convert audio to letter using your image classification model"""
        print(f"🎯 Processing: {Path(audio_path).name}")
        
        try:
            # 1. Convert audio to spectrogram image
            spectrogram_img = self.audio_to_spectrogram_image(audio_path)
            
            # 2. Apply transformations (SAME as training)
            input_tensor = self.transform(spectrogram_img)
            input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
            input_tensor = input_tensor.to(self.device)
            
            print(f"📊 Input shape: {input_tensor.shape}")
            
            # 3. Model prediction
            with torch.no_grad():
                output = self.model(input_tensor)
            
            print(f"📈 Output shape: {output.shape}")
            
            # 4. Get predicted letter
            probabilities = torch.nn.functional.softmax(output, dim=1)
            _, predicted_idx = torch.max(probabilities, 1)
            predicted_class = predicted_idx.item()
            
            # 5. Map to Qubee letter
            # Your model was trained on 26 classes (a-z)
            qubee_letters = "abcdefghijklmnopqrstuvwxyz"
            
            if 0 <= predicted_class < len(qubee_letters):
                predicted_letter = qubee_letters[predicted_class]
                print(f"✅ Predicted letter: '{predicted_letter}' (class {predicted_class})")
                return predicted_letter
            else:
                print(f"⚠️ Invalid class index: {predicted_class}")
                return "?"
            
        except Exception as e:
            error_msg = f"Prediction error: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg
    
    def predict_sequence(self, audio_path, segment_duration=1.0):
        """
        For word prediction: split audio into segments and predict each letter
        segment_duration: seconds per letter (adjust based on your data)
        """
        print(f"🎯 Processing sequence: {Path(audio_path).name}")
        
        try:
            # Load full audio
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            
            # Calculate segment size in samples
            segment_samples = int(segment_duration * sr)
            total_segments = len(audio) // segment_samples
            
            if total_segments == 0:
                print("⚠️ Audio too short for segmentation")
                return self.predict(audio_path)  # Fallback to single prediction
            
            print(f"📊 Splitting into {total_segments} segments ({segment_duration}s each)")
            
            predicted_letters = []
            
            for i in range(total_segments):
                start_sample = i * segment_samples
                end_sample = start_sample + segment_samples
                segment = audio[start_sample:end_sample]
                
                if len(segment) < segment_samples:
                    break  # Last segment might be too short
                
                # Save segment to temp file
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    import soundfile as sf
                    sf.write(f.name, segment, sr)
                    temp_path = f.name
                
                try:
                    # Predict letter for this segment
                    letter = self.predict(temp_path)
                    predicted_letters.append(letter)
                    print(f"  Segment {i+1}/{total_segments}: '{letter}'")
                finally:
                    # Cleanup temp file
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
            
            # Combine letters into word
            word = ''.join(predicted_letters)
            print(f"✅ Predicted word: '{word}'")
            return word
            
        except Exception as e:
            error_msg = f"Sequence prediction error: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg
    
    def decode_output(self, output):
        """Legacy method for compatibility"""
        probabilities = torch.nn.functional.softmax(output, dim=1)
        _, predicted_idx = torch.max(probabilities, 1)
        predicted_class = predicted_idx.item()
        
        qubee_letters = "abcdefghijklmnopqrstuvwxyz"
        if 0 <= predicted_class < len(qubee_letters):
            return qubee_letters[predicted_class]
        return "?"