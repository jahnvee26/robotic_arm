#!/usr/bin/env python3

from octo.model.octo_model import OctoModel
import json
import subprocess
import jax
import jax.numpy as jnp
import copy
from octo.data.utils.data_utils import NormalizationType
import numpy as np

def main():
    print("Loading Octo model...")
    
    # Load Octo model (small)
    model = OctoModel.load_pretrained(
        "/home/welgpu/jahnvee/ddp/octo_main_ws/octo_models/octo-small-1.5"
    )
    
    print("Octo model loaded successfully!")
    print(f"Model type: {type(model)}")
    
    # Configuration
    WINDOW_SIZE = 2
    NUM_STEPS = 5  # Number of inference steps to run
    
    print(f"\n=== Running Inference Loop with WINDOW_SIZE = {WINDOW_SIZE} ===")
    
    # Create goal image (dummy for now - in real use this would be target state)
    goal_image = jnp.zeros((256, 256, 3))  # Target image
    
    # Language instruction for the task
    language_instruction = "pick up the cube"
    
    try:
        # Initialize dummy images for the window
        # In real use, these would be actual camera observations
        images = []
        for i in range(NUM_STEPS):
            # Create dummy image for each step
            img = jnp.zeros((256, 256, 3))  # RGB image
            images.append(img)
        
        # Run inference loop, this model only uses 3rd person image observations for bridge
        # Collect predicted and true actions
        pred_actions, true_actions = [], []
        
        for step in range(len(images) - (WINDOW_SIZE - 1)):
            print(f"\n--- Inference Step {step + 1} ---")
            
            # Create input window of images
            input_images = np.stack(images[step:step+WINDOW_SIZE])[None]  # Add batch dim
            print(f"Input images shape: {input_images.shape}")
            
            observation = {
                'image_primary': input_images,
                'timestep_pad_mask': np.full((1, input_images.shape[1]), True, dtype=bool)
            }
            
            print(f"Observation image_primary shape: {observation['image_primary'].shape}")
            print(f"Timestep pad mask shape: {observation['timestep_pad_mask'].shape}")
            
            # Create task for this step
            task = model.create_tasks(texts=["pick up the cube"])
            
            # This returns *normalized* actions --> we need to unnormalize using the dataset statistics
            try:
                actions = model.sample_actions(
                    observation,
                    task,
                    unnormalization_statistics=model.dataset_statistics["bridge_dataset"]["action"],
                    rng=jax.random.PRNGKey(step)  # Different random key for each step
                )
                print(f"✅ Inference successful for step {step + 1}")
                
            except Exception as e:
                print(f"Error with unnormalization: {e}")
                print("Trying without unnormalization...")
                actions = model.sample_actions(
                    observation,
                    task,
                    rng=jax.random.PRNGKey(step)
                )
                print(f"✅ Inference successful (no unnormalization) for step {step + 1}")
            
            # Remove batch dimension for processing
            actions = actions[0]  # Remove batch dim
            print(f"Actions shape after removing batch: {actions.shape}")
            print(f"Raw actions: {actions}")
            
            # Store predicted actions
            pred_actions.append(actions)
            
            # Get the final window step for processing 
            final_window_step = step + WINDOW_SIZE - 1
            
            # In a real scenario, you would process the action here
            # For example, extract position and orientation:
            if actions.shape[-1] >= 6:  # Ensure we have at least 6 DOF
                position = actions[:3] if len(actions.shape) == 1 else actions[final_window_step][:3]
                orientation = actions[3:6] if len(actions.shape) == 1 else actions[final_window_step][3:6]
                
                print(f"  Position: [{position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}]")
                print(f"  Orientation: [{orientation[0]:.4f}, {orientation[1]:.4f}, {orientation[2]:.4f}]")
                
                if actions.shape[-1] >= 7:
                    gripper = actions[6] if len(actions.shape) == 1 else actions[final_window_step][6]
                    print(f"  Gripper: {gripper:.4f}")
            else:
                print(f"  Action vector: {actions}")
        
        print(f"\n✅ Completed inference loop with {len(pred_actions)} predictions")
        print(f"Total predictions collected: {len(pred_actions)}")
        
        # Summary of all predictions
        print(f"\n=== Prediction Summary ===")
        for i, action in enumerate(pred_actions):
            print(f"Step {i+1}: {action}")
            
    except Exception as e:
        print(f"Error during inference loop: {e}")
        
        # Fallback to simple single-step inference
        print("\nFalling back to simple inference test...")
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
            
            # Run model inference using the correct sample_actions method
            action = model.sample_actions(
                dummy_obs, 
                dummy_task, 
                unnormalization_statistics=model.dataset_statistics["bridge_dataset"]["action"], 
                rng=jax.random.PRNGKey(0)
            )
            print("✅ Fallback inference successful!")
            output = action
            
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            return
        
        print(f"Final fallback output: {output}")
        
    print("\nInference loop completed. You can now use this pattern for robotic arm control.")

if __name__ == "__main__":
    main()