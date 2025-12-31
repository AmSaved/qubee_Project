import torch
import torch.nn as nn
import librosa
import numpy as np
import os
import sys
from pathlib import Path

# Define your model architecture HERE
class QubeeVoiceModel(nn.Module):
    def __init__(self, num_chars=29):  # 26 letters + space + apostrophe + ?
        super().__init__()
        # CNN layers for audio features
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # LSTM for sequence
        self.lstm = nn.LSTM(128 * 10, 256, bidirectional=True, batch_first=True)  # Adjust based on your input
        
        # Output layer
        self.fc = nn.Linear(256 * 2, num_chars)
    
    def forward(self, x):
        # x shape: [batch, 1, n_mels, time]
        x = self.conv(x)  # [batch, 128, n_mels/8, time/8]
        
        # Reshape for LSTM
        batch, channels, height, time = x.shape
        x = x.permute(0, 3, 1, 2)  # [batch, time, channels, height]
        x = x.reshape(batch, time, channels * height)  # [batch, time, features]
        
        # LSTM
        x, _ = self.lstm(x)  # [batch, time, hidden*2]
        
        # Output
        x = self.fc(x)  # [batch, time, num_chars]
        return x

class VoiceToTextModel:
    def __init__(self):
        print("=== Loading Qubee Voice Model ===")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")
        
        # Load model
        self.model = self.load_model()
        print("✅ Model loaded successfully")
    
    def load_model(self):
        model_path = Path("media/models/qubee_voice_model.pth")
        
        if not model_path.exists():
            print(f"❌ ERROR: Model not found at {model_path}")
            print(f"Place your trained model file at: {model_path.absolute()}")
            sys.exit(1)
        
        print(f"📦 Loading model weights from: {model_path}")
        
        try:
            # Load state_dict (weights)
            state_dict = torch.load(str(model_path), map_location=self.device)
            
            # Create model instance
            model = QubeeVoiceModel(num_chars=29)  # Adjust num_chars as needed
            
            # Load weights into model
            model.load_state_dict(state_dict)
            
            # Move to device and set eval mode
            model = model.to(self.device)
            model.eval()
            
            print(f"✅ Model weights loaded successfully")
            print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            return model
            
        except Exception as e:
            print(f"❌ ERROR loading model: {e}")
            print("\n🔧 If this fails, your model might need different architecture.")
            print("   Adjust the QubeeVoiceModel class above to match your training.")
            sys.exit(1)
    
    def preprocess_audio(self, audio_path):
        """Preprocess audio to match your training"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            
            # Trim silence
            audio, _ = librosa.effects.trim(audio, top_db=20)
            
            # Ensure minimum length
            if len(audio) < sr:
                padding = sr - len(audio)
                audio = np.pad(audio, (0, padding), mode='constant')
            
            # Create mel spectrogram
            mel = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_mels=80,
                n_fft=400,
                hop_length=160,
                fmin=0,
                fmax=8000
            )
            
            # Convert to log scale
            log_mel = librosa.power_to_db(mel)
            
            # Normalize
            log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-10)
            
            return log_mel
            
        except Exception as e:
            print(f"❌ Audio preprocessing error: {e}")
            raise
    
    def predict(self, audio_path):
        """Convert audio to text"""
        print(f"🎯 Processing: {Path(audio_path).name}")
        
        try:
            # Preprocess audio
            features = self.preprocess_audio(audio_path)
            
            # Convert to tensor [1, 1, n_mels, time]
            input_tensor = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0)
            input_tensor = input_tensor.to(self.device)
            
            print(f"📊 Input shape: {input_tensor.shape}")
            
            # Model prediction
            with torch.no_grad():
                output = self.model(input_tensor)
            
            print(f"📈 Output shape: {output.shape}")
            
            # Decode output to text
            text = self.decode_output(output)
            
            print(f"✅ Prediction: '{text}'")
            return text
            
        except Exception as e:
            error_msg = f"Prediction error: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg
    
    def decode_output(self, output):
        """Convert model output to Qubee text"""
        # Get character probabilities
        probs = torch.softmax(output, dim=-1)  # Convert to probabilities
        
        # Get most likely character at each time step
        _, char_indices = torch.max(probs, dim=-1)
        
        # Qubee alphabet (Afan Oromo)
        qubee_alphabet = " abcdefghijklmnopqrstuvwxyz'"  # Add more characters if needed
        
        # Convert indices to characters
        char_list = []
        for idx in char_indices[0].cpu().numpy():
            if idx < len(qubee_alphabet):
                char_list.append(qubee_alphabet[idx])
        
        # Join characters
        text = ''.join(char_list)
        
        # Clean up: remove repeated characters and trim
        text = ''.join([c for i, c in enumerate(text) if i == 0 or c != text[i-1]])
        text = text.strip()
        
        # If text is empty, return placeholder
        if not text:
            text = "(No text recognized)"
        
        return text