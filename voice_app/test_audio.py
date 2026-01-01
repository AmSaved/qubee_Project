import os
import sys
sys.path.append('.')
from voice_app.ml_model import VoiceToTextModel

model = VoiceToTextModel()

# Test with a short audio file
test_file = "test.wav"  # Create a 1-second test recording
if os.path.exists(test_file):
    result = model.predict(test_file)
    print(f"Single prediction: {result}")
    
#    result2 = model.predict_sequence(test_file)
#    print(f"Sequence prediction: {result2}")
else:
    print("Create a test.wav file first")