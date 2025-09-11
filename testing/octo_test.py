#!/usr/bin/env python3

"""
Octo Model Test Script

This script loads and tests the Octo model for robotic arm control.
"""

from octo.model.octo_model import OctoModel
import json
import subprocess
import jax
import jax.numpy as jnp
import copy
from octo.data.utils.data_utils import NormalizationType
import numpy as np

def main():
    """
    Load and test the Octo model
    """
    print("Loading Octo model...")
    
    # Load Octo model (small)
    model = OctoModel.load_pretrained(
        "/home/welgpu/jahnvee/ddp/octo_main_ws/octo_models/octo-small-1.5"
    )
    
    print("Octo model loaded successfully!")
    print(f"Model type: {type(model)}")
    
    # Print model information
    print("\nModel configuration:")
    if hasattr(model, 'config'):
        print(f"Config: {model.config}")
    
    # Test basic model functionality
    print("\nTesting model...")
    # Example: Create dummy observation and task
    # You can modify these based on your specific use case
    try:
        # Create a test observation with correct format (batch + time horizon)
        img = jnp.zeros((256, 256, 3))  # Example image shape
        # Add batch + time horizon dimensions [batch, time, height, width, channels]
        img = img[np.newaxis, np.newaxis, ...]  # Shape: (1, 1, 256, 256, 3)
        
        dummy_obs = {
            'image_primary': img,
            'timestep_pad_mask': np.array([[True]])  # Shape: (1, 1) - batch x time
        }
        
        # Create a simple test task using model.create_tasks()
        dummy_task = model.create_tasks(texts=["move to target position"])
        
        print("Model loaded and ready for inference!")
        print(f"Observation keys: {list(dummy_obs.keys())}")
        print(f"Image shape: {dummy_obs['image_primary'].shape}")
        print(f"Timestep mask shape: {dummy_obs['timestep_pad_mask'].shape}")
        
        # Test model inference to see what output it gives
        print("\n=== Testing Model Inference ===")
        
        # Run model inference using the correct sample_actions method
        print("Running model inference with sample_actions...")
        try:
            action = model.sample_actions(
                dummy_obs, 
                dummy_task, 
                unnormalization_statistics=model.dataset_statistics["bridge_dataset"]["action"], 
                rng=jax.random.PRNGKey(0)
            )
            print("✅ Model inference successful!")
            output = action  # For compatibility with rest of the code
            
        except Exception as e1:
            print(f"sample_actions failed: {e1}")
            print("Available dataset statistics keys:", list(model.dataset_statistics.keys()) if hasattr(model, 'dataset_statistics') else "No dataset_statistics")
            
            # Try without unnormalization_statistics
            try:
                print("Trying without unnormalization_statistics...")
                action = model.sample_actions(
                    dummy_obs, 
                    dummy_task, 
                    rng=jax.random.PRNGKey(0)
                )
                print("✅ Model inference successful (without unnormalization)!")
                output = action
                
            except Exception as e2:
                print(f"sample_actions also failed without unnormalization: {e2}")
                return
        
        print(f"\nModel Output Type: {type(output)}")
        
        # Handle the action output (could be direct array or dict)
        if hasattr(output, 'shape'):
            # Direct action array
            actions = output
            print(f"\nActions shape: {actions.shape}")
            print(f"Actions dtype: {actions.dtype}")
            print(f"Actions (first few values): {actions.flatten()[:10]}")
            print(f"Actions min: {jnp.min(actions):.6f}")
            print(f"Actions max: {jnp.max(actions):.6f}")
            print(f"Actions mean: {jnp.mean(actions):.6f}")
            print(f"Full actions: {actions}")
            
        elif isinstance(output, dict):
            print(f"Model Output Keys: {list(output.keys())}")
            
            # Print the action output
            if 'actions' in output:
                actions = output['actions']
                print(f"\nActions shape: {actions.shape}")
                print(f"Actions dtype: {actions.dtype}")
                print(f"Actions (first few values): {actions.flatten()[:10]}")
                print(f"Actions min: {jnp.min(actions):.6f}")
                print(f"Actions max: {jnp.max(actions):.6f}")
                print(f"Actions mean: {jnp.mean(actions):.6f}")
            
            # Print all outputs for debugging
            print(f"\nFull model output:")
            for key, value in output.items():
                if hasattr(value, 'shape'):
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                    if value.size <= 20:  # Only print small arrays
                        print(f"    values: {value}")
                    else:
                        print(f"    sample values: {value.flatten()[:5]}...")
                else:
                    print(f"  {key}: {value}")
        else:
            print(f"Unexpected output type: {type(output)}")
            print(f"Output: {output}")
        
        print("\nYou can now use the model for robotic arm control.")
        
    except Exception as e:
        print(f"Error during model testing: {e}")
        print("Model loaded but may need specific input format adjustments.")
        
        # Try alternative inference method
        try:
            print("\nTrying alternative inference method...")
            # Some models might use different method names
            if hasattr(model, 'predict'):
                output = model.predict(dummy_obs, dummy_task)
                print(f"Alternative method output: {output}")
            elif hasattr(model, 'forward'):
                output = model.forward(dummy_obs, dummy_task)
                print(f"Forward method output: {output}")
            else:
                print("Available model methods:")
                methods = [method for method in dir(model) if not method.startswith('_')]
                print(methods[:10])  # Print first 10 methods
                
        except Exception as e2:
            print(f"Alternative method also failed: {e2}")

if __name__ == "__main__":
    main()
