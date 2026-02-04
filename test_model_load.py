#!/usr/bin/env python3
"""
Test script to verify model loading works correctly
"""
import os
import sys
import torch

# Set environment variables
os.environ['HF_REPO_ID'] = 'alghzlee/sepsis-treatment-model'

print('=== Environment Check ===')
print(f'HF_TOKEN set: {"Yes" if os.getenv("HF_TOKEN") else "No"}')
print(f'Device: {torch.device("cuda" if torch.cuda.is_available() else "cpu")}')

# Test importing the model loading function
print('\n=== Import Check ===')
try:
    from app.routes.predict import load_model, device
    print('✓ Successfully imported load_model')
except Exception as e:
    print(f'✗ Import failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test loading the model
print('\n=== Model Loading Test ===')
try:
    model = load_model('app/data/best_agent_ensemble.pt', device)
    if model is not None:
        print('✓ Model loaded successfully!')
        print(f'Model type: {type(model)}')
        print(f'Number of agents: {len(model.agents) if hasattr(model, "agents") else "N/A"}')
    else:
        print('✗ Model loading returned None')
        sys.exit(1)
except Exception as e:
    import traceback
    print(f'✗ Model loading failed: {e}')
    traceback.print_exc()
    sys.exit(1)

print('\n=== All Tests Passed ===')
