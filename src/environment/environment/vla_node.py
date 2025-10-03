#!/usr/bin/env python3

"""
VLA (Vision-Language-Action) Node for Robot Arm

This node uses the Octo VLA model to generate target positions from camera images and text prompts.
It subscribes to camera images and text commands, then publishes target positions.

Subscribes to: 
- /camera/image (sensor_msgs/Image)
- /text_prompt (std_msgs/String)

Publishes to: 
- /target_position (geometry_msgs/Point)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import cv2
import numpy as np
import threading
import time

# Octo model imports
from octo.model.octo_model import OctoModel
import jax
import jax.numpy as jnp

class VLANode(Node):
    """
    ROS2 Node for Vision-Language-Action using Octo model
    
    Subscribes to: /camera/image, /text_prompt
    Publishes to: /target_position
    """

    def __init__(self):
        super().__init__('vla_node')
        
        # Initialize CV bridge for image conversion
        self.bridge = CvBridge()
        
        # Model and processing state
        self.model = None
        self.model_loaded = False
        self.current_image = None
        self.current_prompt = "move to target position"  # Default prompt
        self.processing_lock = threading.Lock()
        
        # Action sequence execution state
        self.action_sequence = None
        self.current_timestep = 0
        self.sequence_length = 0
        
        # Publishers
        self.position_publisher = self.create_publisher(
            Point,
            '/target_position',
            10
        )
        
        # Subscribers
        self.image_subscription = self.create_subscription(
            Image,
            '/camera/image',
            self.image_callback,
            10
        )
        
        self.prompt_subscription = self.create_subscription(
            String,
            '/text_prompt',
            self.prompt_callback,
            10
        )
        
        # Processing timer (run inference at lower frequency)
        self.inference_timer = self.create_timer(2.0, self.run_inference)  # 0.5 Hz for new sequences
        
        # Action execution timer (execute actions from sequence at higher frequency)
        self.execution_timer = self.create_timer(0.5, self.execute_next_action)  # 2 Hz for smooth execution
        
        # Load model in separate thread to avoid blocking
        self.model_thread = threading.Thread(target=self.load_model)
        self.model_thread.daemon =    True
        self.model_thread.start()
        
        self.get_logger().info("VLA Node initialized. Loading Octo model...")

    def load_model(self):
        """Load the Octo VLA model in a separate thread"""
        self.get_logger().info("Loading Octo model...")
        
        # Load Octo model (small)
        self.model = OctoModel.load_pretrained(
            "/home/welgpu/jahnvee/ddp/octo_main_ws/octo_models/octo-small-1.5"
        )
        
        self.model_loaded = True
        self.get_logger().info("✅ Octo model loaded successfully!")
        
        # Log model information
        if hasattr(self.model, 'dataset_statistics'):
            datasets = list(self.model.dataset_statistics.keys())
            self.get_logger().info(f"Available datasets: {datasets}")

    def image_callback(self, msg: Image):
        """Callback for receiving camera images"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            with self.processing_lock:
                self.current_image = cv_image
                
                # Debug information
                self.get_logger().info(
                    f"Received image: {msg.width}x{msg.height}, "
                    f"frame_id: {msg.header.frame_id}, "
                    f"timestamp: {msg.header.stamp.sec}.{msg.header.stamp.nanosec}"
                )
                
                # Log image statistics for verification
                if cv_image is not None:
                    mean_val = np.mean(cv_image)
                    self.get_logger().debug(f"Image stats: mean={mean_val:.2f}, shape={cv_image.shape}")
                    
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def prompt_callback(self, msg: String):
        """Callback for receiving text prompts"""
        with self.processing_lock:
            self.current_prompt = msg.data
            self.get_logger().info(f"Received prompt: '{self.current_prompt}'")

    def preprocess_image(self, cv_image):
        """Preprocess image for Octo model"""
        # Resize to model input size (256x256)
        resized = cv2.resize(cv_image, (256, 256))
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb_image.astype(np.float32) / 255.0
        
        # Convert to JAX array and add batch + time dimensions
        jax_image = jnp.array(normalized)
        jax_image = jax_image[np.newaxis, np.newaxis, ...]  # Shape: (1, 1, 256, 256, 3)
        
        return jax_image

    def create_observation(self, image):
        """Create observation dict for Octo model"""
        observation = {
            'image_primary': image,
            'timestep_pad_mask': np.array([[True]])  # Shape: (1, 1)
        }
        return observation

    def action_to_position(self, action):
        """Convert model action output to target position
        
        Action format: [1, 4, 7] where:
        - 1: batch size
        - 4: number of action steps (timesteps)
        - 7: degrees of freedom [x, y, z, roll, pitch, yaw, gripper]
        """
        
        # Handle JAX array or numpy array
        if hasattr(action, 'shape'):
            self.get_logger().debug(f"Action shape: {action.shape}")
            
            # Expected format: [1, 4, 7] - [batch, timesteps, dof]
            if len(action.shape) == 3 and action.shape == (1, 4, 7):
                # Store the entire action sequence for sequential execution
                # action[0, :, :] gives us all timesteps: shape [4, 7]
                self.action_sequence = action[0, :, :].copy()  # Shape: [4, 7] 
                self.sequence_length = action.shape[1]  # 4 timesteps
                self.current_timestep = 0  # Start from first timestep
                
                self.get_logger().info(f"Stored new action sequence: {self.sequence_length} timesteps")
                return None  # Don't return a position here, let execute_next_action handle it
                
            elif len(action.shape) == 3:
                # Handle other 3D shapes - store entire sequence
                batch, timesteps, dof = action.shape
                self.action_sequence = action[0, :, :].copy()  # Shape: [timesteps, dof]
                self.sequence_length = timesteps
                self.current_timestep = 0
                self.get_logger().info(f"Stored action sequence from shape [{batch}, {timesteps}, {dof}]")
                return None
                
            elif len(action.shape) == 2:
                # Handle 2D case [batch, dof] or [timesteps, dof]
                action_vector = action[-1, :] if action.shape[0] > 1 else action[0, :]
                
            elif len(action.shape) == 1:
                # Handle 1D case [dof]
                action_vector = action
                
            else:
                self.get_logger().warn(f"Unexpected action shape: {action.shape}")
                action_vector = action.flatten()
                
        else:
            # Convert to numpy if not already
            action_vector = np.array(action).flatten()
        
    
        x, y, z = float(action_vector[0]), float(action_vector[1]), float(action_vector[2])
        
        self.get_logger().debug(f"Raw action values: x={x:.4f}, y={y:.4f}, z={z:.4f}")
        
        # Convert to proper scale and bounds
        # Assuming action is normalized to [-1, 1], scale to robot workspace
        x = float(0.2 + 0.15 * x)  # Scale to [0.05, 0.35] range
        y = float(0.2 * y)         # Scale to [-0.2, 0.2] range  
        z = float(0.15 + 0.1 * z)  # Scale to [0.05, 0.25] range
        
        # Clamp values to safe workspace limits
        x = max(0.05, min(0.35, x))
        y = max(-0.2, min(0.2, y))
        z = max(0.05, min(0.25, z))
        
        return x, y, z

    def execute_next_action(self):
        """Execute the next action in the current sequence"""
        with self.processing_lock:
            # Check if we have a valid action sequence
            if (self.action_sequence is None or 
                self.current_timestep >= self.sequence_length):
                return  # sequence completed
            
            # Get the current timestep action
            action_vector = self.action_sequence[self.current_timestep, :]  # Shape: [7]
            
            # Convert action to target position using existing method
            x, y, z = self.action_vector_to_position(action_vector)
            
            # Create and publish target position
            target_pos = Point()
            target_pos.x = x
            target_pos.y = y  
            target_pos.z = z
            
            self.position_publisher.publish(target_pos)
            
            self.get_logger().info(
                f"Executed timestep {self.current_timestep + 1}/{self.sequence_length}: "
                f"x={x:.3f}, y={y:.3f}, z={z:.3f}"
            )
            
            # Move to next timestep
            self.current_timestep += 1
            
            # Check if sequence is complete
            if self.current_timestep >= self.sequence_length:
                self.get_logger().info("✅ Action sequence completed!")
                self.action_sequence = None  # Clear completed sequence

    def action_vector_to_position(self, action_vector):
        """Convert a single action vector to target position (helper method)"""
        # Extract only x, y, z coordinates (first 3 elements of the 7-DOF vector)
        if len(action_vector) >= 3:
            x, y, z = float(action_vector[0]), float(action_vector[1]), float(action_vector[2])
            
            # Log all available action components for debugging
            if len(action_vector) >= 7:
                roll, pitch, yaw, gripper = action_vector[3:7]
                self.get_logger().debug(
                    f"Action: x={x:.4f}, y={y:.4f}, z={z:.4f}, "
                    f"roll={roll:.4f}, pitch={pitch:.4f}, yaw={yaw:.4f} (unused), gripper={gripper:.4f}"
                )
            
            # Convert to proper scale and bounds
            x = float(0.2 + 0.15 * x)  # Scale to [0.05, 0.35] range
            y = float(0.2 * y)         # Scale to [-0.2, 0.2] range  
            z = float(0.15 + 0.1 * z)  # Scale to [0.05, 0.25] range
            
            # Clamp values to safe workspace limits
            x = max(0.05, min(0.35, x))
            y = max(-0.2, min(0.2, y))
            z = max(0.05, min(0.25, z))
            
        else:
            # Default position if action is too short
            x, y, z = 0.2, 0.0, 0.1
            self.get_logger().warn(f"Action vector too short ({len(action_vector)}), using default position")

    def run_inference(self):
        """Run VLA inference to generate new action sequence"""
        if not self.model_loaded or self.model is None:
            return
        
        # Skip if we're still executing a previous sequence
        with self.processing_lock:
            if (self.action_sequence is not None and 
                self.current_timestep < self.sequence_length):
                self.get_logger().debug(f"Still executing sequence: {self.current_timestep}/{self.sequence_length}")
                return
            
            current_image = self.current_image
            current_prompt = self.current_prompt
        
        if current_image is None:
            self.get_logger().warn("No image available for inference")
            return
        
        try:
            # Preprocess image
            processed_image = self.preprocess_image(current_image)
            
            # Create observation
            observation = self.create_observation(processed_image)
            
            # Create task from prompt
            task = self.model.create_tasks(texts=[current_prompt])
            
            # Run model inference to get new action sequence
            action = self.model.sample_actions(
                observation, 
                task, 
                unnormalization_statistics=self.model.dataset_statistics["bridge_dataset"]["action"], 
                rng=jax.random.PRNGKey(int(time.time() * 1000) % 2**31)
            )
            
            # Store the action sequence for sequential execution
            # This will set self.action_sequence and reset self.current_timestep
            self.action_to_position(action)
            
            self.get_logger().info("Generated new action sequence for execution")
            self.get_logger().debug(f"Raw action shape: {action.shape if hasattr(action, 'shape') else 'no shape'}")
            
        except Exception as e:
            self.get_logger().error(f"Error during VLA inference: {e}")


def main(args=None):
    rclpy.init(args=args)
    
    vla_node = VLANode()
    rclpy.spin(vla_node)

    vla_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
