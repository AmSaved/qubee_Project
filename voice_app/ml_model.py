import torch
import librosa
import numpy as np
import os
from django.conf import settings

class VoiceToTextModel:
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model()
    
    def load_model(self):
        """Load your PyTorch model"""
        model_path = os.path.join(settings.MEDIA_ROOT, 'models', 'qubee_voice_model.pth')
        
        if not os.path.exists(model_path):
            print(f"⚠️ Place your model at: {model_path}")
            print("Using test mode for now")
            return
        
        try:
            # Load your actual model
            self.model = torch.load(model_path, map_location=self.device)
            if hasattr(self.model, 'eval'):
                self.model.eval()
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Using test mode")
    
    def preprocess_audio(self, audio_path):
        """Convert audio to features"""
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Trim silence
        audio, _ = librosa.effects.trim(audio, top_db=20)
        
        # Create mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=80, n_fft=400, hop_length=160
        )
        log_mel = librosa.power_to_db(mel)
        
        # Normalize
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-10)
        
        # Convert to tensor
        tensor = torch.FloatTensor(log_mel).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)
    
    def predict(self, audio_path):
        """Convert audio to text"""
        # If no model, return test text
        if self.model is None:
            test_texts = [
                "akkam jirtu",
                "galatoomaa",
                "nagaa",
                "dhaabbatanii",
                "maaloo"
            ]
            import random
            return random.choice(test_texts)
        
        try:
            # Preprocess
            features = self.preprocess_audio(audio_path)
            
            # Predict
            with torch.no_grad():
                output = self.model(features)
            
            # Convert to text (adjust based on your model output)
            # This is a placeholder - you need to adapt this to your model
            text = self._decode_output(output)
            return text
            
        except Exception as e:
            return f"Error: {str(e)[:50]}"
    
    def _decode_output(self, output):
        """Convert model output to text - MODIFY THIS FOR YOUR MODEL"""
        # Example decoding - you MUST change this for your actual model
        
        # If output is tensor with character probabilities
        if isinstance(output, torch.Tensor):
            # Get character indices
            _, indices = torch.max(output, dim=-1)
            
            # Map to Qubee alphabet
            qubee_chars = " abcdefghijklmnopqrstuvwxyz'ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            text = ''.join([qubee_chars[i] for i in indices[0] if i < len(qubee_chars)])
            
            # Remove repeated characters
            text = ''.join([char for i, char in enumerate(text) 
                          if i == 0 or char != text[i-1]])
            return text.strip()
        
        return "Your text here"