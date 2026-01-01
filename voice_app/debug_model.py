import torch
import sys
import os

sys.path.append('.')
model_path = "media/models/qubee_voice_model.pth"

print(f"Loading model from: {model_path}")
checkpoint = torch.load(model_path, map_location='cpu')

print(f"\n=== MODEL DEBUG INFO ===")
print(f"Type: {type(checkpoint)}")

if isinstance(checkpoint, torch.nn.Module):
    print("✅ It's a PyTorch Model")
    print(f"Model architecture: {checkpoint}")
    
    # Check output layer
    if hasattr(checkpoint, 'fc'):
        print(f"FC layer: {checkpoint.fc}")
        print(f"Output features: {checkpoint.fc.out_features}")
    
elif isinstance(checkpoint, dict):
    print("✅ It's a dictionary")
    print(f"Keys: {list(checkpoint.keys())}")
    
    # Check for state_dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print(f"State dict keys (first 5): {list(state_dict.keys())[:5]}")
        
        # Find output layer
        for key in state_dict.keys():
            if 'fc' in key or 'classifier' in key:
                print(f"Found classifier layer: {key}")
                print(f"Shape: {state_dict[key].shape}")
    
elif isinstance(checkpoint, list):
    print("❓ It's a list")
    print(f"Length: {len(checkpoint)}")
    
else:
    print(f"❓ Unknown type: {type(checkpoint)}")

# Try to make a prediction
print(f"\n=== TEST PREDICTION ===")
try:
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    if isinstance(checkpoint, torch.nn.Module):
        checkpoint.eval()
        with torch.no_grad():
            output = checkpoint(dummy_input)
        print(f"Output shape: {output.shape}")
        print(f"Output sample: {output[0, :5]}")
        
        # Get prediction
        _, pred = torch.max(output, 1)
        print(f"Predicted class index: {pred.item()}")
        
except Exception as e:
    print(f"Prediction failed: {e}")