#!/usr/bin/env python3

"""
Standalone Octo VLA Script (No ROS2 Dependencies)

This script runs the Octo VLA model in the conda environment without ROS2.
It can be called from external scripts or used for testing.

Usage:
    python3 octo_inference.py --image_path /path/to/image.jpg --prompt "move to target"
    OR
    python3 octo_inference.py  # Uses default test image and prompt
"""

import argparse
import numpy as np
import cv2
import time
import sys
import os

# Octo model imports
from octo.model.octo_model import OctoModel
import jax
import jax.numpy as jnp

class OctoInference:

    def __init__(self, model_path="/home/welgpu/jahnvee/ddp/octo_main_ws/octo_models/octo-small-1.5"):
        self.model_path = model_path
        self.model = None
        self.model_loaded = False
        self.WINDOW_SIZE = window_size
        self.frame_buffer = [] #rolling buffer for latest frames
        
        self.load_model()
    
    def load_model(self):
        """Load the Octo VLA model"""
        print(f" Loading Octo model from: {self.model_path}")
        
        self.model = OctoModel.load_pretrained(self.model_path)
        self.model_loaded = True
        print("✅ Octo model loaded successfully!")
        
        # Log model information
        if hasattr(self.model, 'dataset_statistics'):
            datasets = list(self.model.dataset_statistics.keys())
            print(f"Available datasets: {datasets}")
                
    def preprocess_image(self, image_path_or_array):
        """Preprocess image for Octo model
        Args:
            image_path_or_array: Either path to image file or numpy array
        Returns:
            JAX array ready for model input
        """
        # Load image if path provided
        if isinstance(image_path_or_array, str):
            if not os.path.exists(image_path_or_array):
                raise FileNotFoundError(f"Image not found: {image_path_or_array}")
            cv_image = cv2.imread(image_path_or_array)
            if cv_image is None:
                raise ValueError(f"Could not load image: {image_path_or_array}")
        else:
            cv_image = image_path_or_array
        
        resized = cv2.resize(cv_image, (256, 256))
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb_image.astype(np.float32) / 255.0
        
        #need to check if jax_array is needed
        # Convert to JAX array and add batch + time dimensions
        jax_image = jnp.array(normalized)
        jax_image = jax_image[np.newaxis, np.newaxis, ...]  # Shape: (1, 1, 256, 256, 3)
        
        return jax_image
    
    def create_observation(self, image):
        observation = {
            'image_primary': image,
            'timestep_pad_mask': np.ones((1, self.WINDOW_SIZE), dtype=bool),  # Shape: (1, 1)
        }
        return observation
    
    def run_inference(self, image_input, text_prompt="pick up the cube"):
        """Run VLA inference
        Args:
            image_input: Image path (string) or numpy array
            text_prompt: Text instruction for the robot
        Returns:
            dict with action sequence and target positions
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded!")
        
        print(f" Running inference with prompt: '{text_prompt}'")
        
        # Preprocess image
        processed_image = self.preprocess_image(image_input)
        self.frame_buffer.append(processed_image)
        if len(self.frame_buffer) > self.WINDOW_SIZE:
            self.frame_buffer.pop(0)
        if len(self.frame_buffer) < self.WINDOW_SIZE:
            print("waiting for buffer to fill..")
            return None

        image_stack = np.stack(self.frame_buffer, axis=0)  # Shape: (1, WINDOW_SIZE, 256, 256, 3)
        observation = self.create_observation(image_stack)
        task = self.model.create_tasks(texts=[text_prompt])
        
        # Run model inference
        print(" Running Octo model inference...")
        start_time = time.time()
        
        action = self.model.sample_actions(
            observation, 
            task, 
            unnormalization_statistics=self.model.dataset_statistics["bridge_dataset"]["action"], 
            rng=jax.random.PRNGKey(int(time.time() * 1000) % 2**31)
        )
        
        inference_time = time.time() - start_time
        print(f"⚡ Inference completed in {inference_time:.3f}s")
        
        # Process results
        result = self.process_action_sequence(action)
        
        return result
    
    def process_action_sequence(self, action):
        """Process action sequence into readable format
        Args:
            action: Raw action from Octo model
        Returns:
            dict with processed action sequence and target positions
        """
        result = {
            'raw_action_shape': action.shape if hasattr(action, 'shape') else 'no shape',
            'action_sequence': [],
            'target_positions': [],
            'timestep_count': 0
        }
        
        # Handle different action shapes
        if hasattr(action, 'shape'):
            print(f" Action shape: {action.shape}")
            
            if len(action.shape) == 3 and action.shape[0] == 1:  # [1, timesteps, dof]
                batch, timesteps, dof = action.shape
                result['timestep_count'] = timesteps
                
                print(f" Processing {timesteps} timesteps with {dof} DOF each")
                
                # Process each timestep
                for t in range(timesteps):
                    action_vector = action[0, t, :]  # Get action for timestep t
                    
                    # Convert to target position
                    position = self.action_to_position(action_vector)
                    
                    timestep_data = {
                        'timestep': t,
                        'raw_action': action_vector.tolist() if hasattr(action_vector, 'tolist') else list(action_vector),
                        'target_position': position
                    }
                    
                    result['action_sequence'].append(timestep_data)
                    result['target_positions'].append(position)
                    
                    print(f"  Timestep {t+1}: x={position['x']:.3f}, y={position['y']:.3f}, z={position['z']:.3f}, "
                          f"roll={position['roll']:.3f}, pitch={position['pitch']:.3f}, yaw={position['yaw']:.3f}")
            
            else:
                # Handle other shapes
                action_flat = action.flatten() if hasattr(action, 'flatten') else np.array(action).flatten()
                if len(action_flat) >= 3:
                    position = self.action_to_position(action_flat)
                    result['target_positions'].append(position)
                    result['action_sequence'].append({
                        'timestep': 0,
                        'raw_action': action_flat.tolist(),
                        'target_position': position
                    })
        
        return result
    
    def action_to_position(self, action_vector):
        """Convert action vector to target position with orientation
        Args:
            action_vector: Single action vector [x, y, z, roll, pitch, yaw, gripper]
        Returns:
            dict with scaled target position
        """
        # Extract x, y, z coordinates (first 3 elements)
        if len(action_vector) >= 3:
            x, y, z = float(action_vector[0]), float(action_vector[1]), float(action_vector[2])
            
            # Extract orientation (elements 3-5) if available
            roll, pitch, yaw = 0.0, 0.0, 0.0
            if len(action_vector) >= 6:
                roll, pitch, yaw = float(action_vector[3]), float(action_vector[4]), float(action_vector[5])
            
            # Extract gripper if available
            gripper = 0.0
            if len(action_vector) >= 7:
                gripper = float(action_vector[6])
                
            # Log all DOF for debugging
            print(f"    Raw action: x={x:.3f}, y={y:.3f}, z={z:.3f}, "
                  f"roll={roll:.3f}, pitch={pitch:.3f}, yaw={yaw:.3f}, gripper={gripper:.3f}")
            
            # Convert position to proper scale and bounds (same as VLA node)
            x = float(0.2 + 0.15 * x)  # Scale to [0.05, 0.35] range
            y = float(0.2 * y)         # Scale to [-0.2, 0.2] range  
            z = float(0.15 + 0.1 * z)  # Scale to [0.05, 0.25] range
            
            # Clamp position values to safe workspace limits
            x = max(0.05, min(0.35, x))
            y = max(-0.2, min(0.2, y))
            z = max(0.05, min(0.25, z))
            
            # Scale orientation values (keep as delta angles in radians)
            # Note: These are typically small delta values, so minimal scaling
            roll = max(-0.5, min(0.5, roll))    # Limit to ±0.5 radians (~±30 degrees)
            pitch = max(-0.5, min(0.5, pitch))
            yaw = max(-0.5, min(0.5, yaw))
            
        else:
            # Default position if action is too short
            x, y, z = 0.2, 0.0, 0.1
            roll, pitch, yaw = 0.0, 0.0, 0.0
            print(f"⚠️ Action vector too short ({len(action_vector)}), using default position")
        
        return {
            'x': x, 
            'y': y, 
            'z': z,
            'roll': roll,
            'pitch': pitch,
            'yaw': yaw
        }


def main():
    """Main function for standalone usage"""
    parser = argparse.ArgumentParser(description='Standalone Octo VLA Inference')
    parser.add_argument('--image_path', type=str, help='Path to input image')
    parser.add_argument('--prompt', type=str, default='move to target position', 
                       help='Text prompt for the robot')
    parser.add_argument('--model_path', type=str, 
                       default='/home/welgpu/jahnvee/ddp/octo_main_ws/octo_models/octo-small-1.5',
                       help='Path to Octo model')
    
    args = parser.parse_args()
    
    # Initialize Octo inference
    octo = OctoInference(model_path=args.model_path)
    
    # Use provided image 
    image_input = args.image_path
    print(f" Using image: {args.image_path}")

    # Run inference
    result = octo.run_inference(image_input, args.prompt)
    
    # Print results
    print("\n" + "="*50)
    print(" OCTO VLA INFERENCE RESULTS")
    print("="*50)
    print(f" Action shape: {result['raw_action_shape']}")
    print(f"Timesteps: {result['timestep_count']}")
    print(f" Target positions:")
    
    for i, pos in enumerate(result['target_positions']):
        print(f"  Position {i+1}: x={pos['x']:.3f}, y={pos['y']:.3f}, z={pos['z']:.3f}")
    
    # Save result for potential use by other scripts
    import json
    output_file = '/tmp/octo_result.json'
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f" Results saved to: {output_file}")
        
  

if __name__ == '__main__':
    main()