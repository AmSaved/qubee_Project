# test_final.py
import sys
sys.path.append('.')
from voice_app.ml_model import VoiceToTextModel

print("=== FINAL TEST ===")
model = VoiceToTextModel()

# Test with a real recording
import numpy as np
import soundfile as sf
import tempfile
import os

# Create a simple test: sine wave at 440Hz (letter 'A' frequency)
duration = 1.0  # 1 second
sr = 16000
t = np.linspace(0, duration, int(sr * duration))
test_audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440Hz = musical note A

with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
    sf.write(f.name, test_audio, sr)
    temp_path = f.name

try:
    result = model.predict(temp_path)
    print(f"\n🎯 FINAL RESULT: '{result}'")
finally:
    os.unlink(temp_path)