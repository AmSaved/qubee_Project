import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
import os
import sys
from pathlib import Path

# ============================================
# RESNET MODEL ARCHITECTURE (Matches your model)
# ============================================

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=26):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # Initial convolution
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # ResNet layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layer - OUTPUT: 26 classes (letters)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # Input: [batch, 1, n_mels, time]
        out = F.relu(self.bn1(self.conv1(x)))
        
        # ResNet blocks
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        # Pooling
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        
        # Classification
        out = self.fc(out)
        return out

def ResNet18(num_classes=26):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

def ResNet34(num_classes=26):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)

# ============================================
# VOICE TO TEXT MODEL
# ============================================

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
            # Load state_dict
            state_dict = torch.load(str(model_path), map_location=self.device)
            
            # Create ResNet18 model (matches your architecture)
            # Your model has 26 output classes (letters)
            model = ResNet18(num_classes=26)
            
            # Load weights
            model.load_state_dict(state_dict)
            
            # Move to device and set eval
            model = model.to(self.device)
            model.eval()
            
            print(f"✅ Model weights loaded successfully")
            print(f"📊 Model: ResNet18 with 26 output classes")
            print(f"📊 Parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            return model
            
        except Exception as e:
            print(f"❌ ERROR loading model: {e}")
            print("\nTrying ResNet34 instead...")
            
            try:
                # Try ResNet34
                model = ResNet34(num_classes=26)
                model.load_state_dict(state_dict)
                model = model.to(self.device)
                model.eval()
                print(f"✅ Success with ResNet34")
                return model
            except:
                print("❌ Both ResNet18 and ResNet34 failed")
                print("Check your model architecture")
                sys.exit(1)
    
    def preprocess_audio(self, audio_path):
        """Preprocess audio to match your training"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            
            # Trim silence
            audio, _ = librosa.effects.trim(audio, top_db=20)
            
            # Ensure minimum length (2 seconds)
            min_length = sr * 2
            if len(audio) < min_length:
                # Pad with silence
                padding = min_length - len(audio)
                audio = np.pad(audio, (0, padding), mode='constant')
            else:
                # Take first 5 seconds max
                max_length = sr * 5
                audio = audio[:max_length]
            
            print(f"📊 Audio length: {len(audio)/sr:.2f}s")
            
            # Create mel spectrogram (80 mels is standard for ResNet audio)
            mel = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_mels=80,          # Standard for audio ResNet
                n_fft=512,          # Standard
                hop_length=256,     # Standard
                fmin=20,
                fmax=8000
            )
            
            # Convert to log scale
            log_mel = librosa.power_to_db(mel)
            
            # Normalize to [0, 1]
            log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-10)
            
            print(f"📊 Mel shape: {log_mel.shape}")
            
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
            
            # Resize to fixed size for ResNet (224x224 is standard, but we use 80xtime)
            # ResNet expects [batch, channels, height, width]
            # We have [n_mels=80, time]
            
            # Pad/trim time dimension to 224 (ResNet standard) or keep original
            target_time = 224  # Standard ResNet input
            
            if features.shape[1] < target_time:
                # Pad
                padding = target_time - features.shape[1]
                features = np.pad(features, ((0, 0), (0, padding)), mode='constant')
            else:
                # Trim
                features = features[:, :target_time]
            
            print(f"📊 Final input shape: {features.shape}")
            
            # Convert to tensor [1, 1, height, width]
            input_tensor = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0)
            input_tensor = input_tensor.to(self.device)
            
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
        # Your model outputs class probabilities for 26 classes
        # Get predicted class
        probabilities = F.softmax(output, dim=1)
        _, predicted_idx = torch.max(probabilities, 1)
        class_idx = predicted_idx.item()
        
        # Map class index to Qubee character
        # Your model was trained with 26 classes (probably letters a-z)
        qubee_alphabet = "abcdefghijklmnopqrstuvwxyz"  # 26 letters
        
        if class_idx < len(qubee_alphabet):
            predicted_char = qubee_alphabet[class_idx]
        else:
            predicted_char = "?"
        
        # Since this is a classification model (not sequence),
        # it predicts one character per audio clip
        # You might need to modify this if your model works differently
        
        return predicted_char
    
    def decode_sequence(self, audio_path):
        """Alternative: If your model is for sequences, split audio and predict each segment"""
        # This is for sequence prediction
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            
            # Split into 1-second segments
            segment_length = sr  # 1 second
            segments = []
            
            for i in range(0, len(audio), segment_length):
                segment = audio[i:i + segment_length]
                if len(segment) == segment_length:
                    segments.append(segment)
            
            print(f"📊 Split into {len(segments)} segments")
            
            # Predict each segment
            predicted_chars = []
            
            for i, segment in enumerate(segments):
                # Save segment to temp file
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    import soundfile as sf
                    sf.write(f.name, segment, sr)
                    temp_path = f.name
                
                # Predict character for this segment
                features = self.preprocess_audio(temp_path)
                
                # Prepare input
                target_time = 224
                if features.shape[1] < target_time:
                    features = np.pad(features, ((0, 0), (0, target_time - features.shape[1])), mode='constant')
                else:
                    features = features[:, :target_time]
                
                input_tensor = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0)
                input_tensor = input_tensor.to(self.device)
                
                with torch.no_grad():
                    output = self.model(input_tensor)
                
                probabilities = F.softmax(output, dim=1)
                _, predicted_idx = torch.max(probabilities, 1)
                class_idx = predicted_idx.item()
                
                qubee_alphabet = "abcdefghijklmnopqrstuvwxyz"
                if class_idx < len(qubee_alphabet):
                    predicted_chars.append(qubee_alphabet[class_idx])
                else:
                    predicted_chars.append("?")
                
                # Cleanup
                os.unlink(temp_path)
            
            # Join characters
            text = ''.join(predicted_chars)
            return text
            
        except Exception as e:
            return f"Sequence error: {str(e)}"