#!/usr/bin/env python3

"""
Octo Bridge Script (ROS2 Node)

This script acts as a bridge between ROS2 and the standalone Octo inference script.

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
import subprocess
import json
import tempfile
import os
import threading
import time

class OctoBridgeNode(Node):
    """
    ROS2 Node that bridges to standalone Octo inference
    """

    def __init__(self):
        super().__init__('octo_bridge_node')
        
        # Initialize CV bridge for image conversion
        self.bridge = CvBridge()
        
        # State variables
        self.current_image = None
        self.current_prompt = "pick up the cube"  # Default prompt
        self.processing_lock = threading.Lock()
        
        # Action sequence execution state (similar to VLA node)
        self.action_sequence = None
        self.current_timestep = 0
        self.sequence_length = 0
        
        # Path to conda environment and Octo script
        self.conda_env = "octo"
        self.octo_script = "/home/welgpu/jahnvee/ddp/robotic_arm/src/environment/environment/octo_inference.py"
        
        # Performance configuration parameters
        self.declare_parameter('inference_rate', 0.5)    # Hz (0.5 Hz = every 2 seconds)
        self.declare_parameter('execution_rate', 2.0)     # Hz (2 Hz = every 0.5 seconds) 
        self.declare_parameter('use_image_caching', True) # Cache identical images
        
        inference_rate = self.get_parameter('inference_rate').value
        execution_rate = self.get_parameter('execution_rate').value
        self.use_caching = self.get_parameter('use_image_caching').value
        
        self.get_logger().info(f"⚙️ Performance config: inference={inference_rate}Hz, execution={execution_rate}Hz, caching={self.use_caching}")
        
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
        
        # Processing timer (run inference at configurable frequency)
        self.inference_timer = self.create_timer(1.0/inference_rate, self.run_inference)
        
        # Action execution timer (execute actions from sequence at higher frequency)
        self.execution_timer = self.create_timer(1.0/execution_rate, self.execute_next_action)
        
        self.get_logger().info("🌉 Octo Bridge Node initialized - connecting ROS2 to Octo conda environment")

    def image_callback(self, msg: Image):
        """Callback for receiving camera images"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            with self.processing_lock:
                self.current_image = cv_image
                
                self.get_logger().debug(
                    f"Received image: {msg.width}x{msg.height}, "
                    f"frame_id: {msg.header.frame_id}"
                )
                
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def prompt_callback(self, msg: String):
        """Callback for receiving text prompts"""
        with self.processing_lock:
            self.current_prompt = msg.data
            self.get_logger().info(f"Received prompt: '{self.current_prompt}'")

    def run_inference(self):
        """Run Octo inference via subprocess to conda environment"""
        start_time = time.time()
        
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
            if self.use_caching:
                # Use shared memory approach for better performance
                import hashlib
                
                # Create persistent temp file path based on image hash for caching
                image_hash = hashlib.md5(current_image.tobytes()).hexdigest()[:8]
                temp_image_path = f'/tmp/octo_camera_{image_hash}.jpg'
                
                # Only save if image changed (avoid unnecessary file I/O)
                if not os.path.exists(temp_image_path):
                    cv2.imwrite(temp_image_path, current_image)
                    self.get_logger().debug(f" Saved new camera image: {temp_image_path}")
                else:
                    self.get_logger().debug(f" Reusing cached image: {temp_image_path}")
            else:
                # Direct temp file approach (no caching)
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_img:
                    cv2.imwrite(tmp_img.name, current_image)
                    temp_image_path = tmp_img.name
                self.get_logger().debug(f" Saved temp image: {temp_image_path}")
            
            self.get_logger().info(f" Running Octo inference: '{current_prompt}'")
            
            # Construct command to run Octo script in conda environment
            cmd = [
                'conda', 'run', '-n', self.conda_env, 'python3', 
                self.octo_script,
                '--image_path', temp_image_path,
                '--prompt', current_prompt
            ]
            
            # Run Octo inference subprocess
            start_time = time.time()
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=120  # 120 sec timeout (2 minutes)
            )
            
            inference_time = time.time() - start_time
            
            # Clean up temporary image
            os.unlink(temp_image_path)
            
            if result.returncode == 0:
                # Read results from JSON file
                result_file = '/tmp/octo_result.json'
                if os.path.exists(result_file):
                    with open(result_file, 'r') as f:
                        octo_result = json.load(f)
                    
                    # Store action sequence for sequential execution
                    self.store_action_sequence(octo_result)
                    
                    self.get_logger().info(f" Octo inference completed in {inference_time:.2f}s")
                    self.get_logger().info(f" Generated {len(octo_result['target_positions'])} target positions")
                else:
                    self.get_logger().error("Octo result file not found")
            else:
                self.get_logger().error(f"Octo inference failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            self.get_logger().error("Octo inference timeout (120s)")
        except Exception as e:
            self.get_logger().error(f"Error running Octo inference: {e}")
        finally:
            # Log total timing performance
            total_time = time.time() - start_time
            self.get_logger().info(f"Total inference cycle: {total_time:.2f}s")

    def store_action_sequence(self, octo_result):
        """Store action sequence from Octo results for sequential execution"""
        with self.processing_lock:
            if 'target_positions' in octo_result and octo_result['target_positions']:
                self.action_sequence = octo_result['target_positions']
                self.sequence_length = len(self.action_sequence)
                self.current_timestep = 0
                
                self.get_logger().info(f"📦 Stored action sequence: {self.sequence_length} timesteps")
            else:
                self.get_logger().warn("No valid target positions in Octo result")

    def execute_next_action(self):
        """Execute the next action in the current sequence"""
        with self.processing_lock:
            # Check if we have a valid action sequence
            if (self.action_sequence is None or 
                self.current_timestep >= self.sequence_length):
                return  # No sequence or sequence completed
            
            # Get the current timestep action
            position_data = self.action_sequence[self.current_timestep]
            
            # Create and publish target position
            target_pos = Point()
            target_pos.x = position_data['x']
            target_pos.y = position_data['y']
            target_pos.z = position_data['z']
            
            self.position_publisher.publish(target_pos)
            
            self.get_logger().info(
                f" Executed timestep {self.current_timestep + 1}/{self.sequence_length}: "
                f"x={target_pos.x:.3f}, y={target_pos.y:.3f}, z={target_pos.z:.3f}"
            )
            
            # Move to next timestep
            self.current_timestep += 1
            
            # Check if sequence is complete
            if self.current_timestep >= self.sequence_length:
                self.get_logger().info("✅ Action sequence completed!")
                self.action_sequence = None  # Clear completed sequence


def main(args=None):
    rclpy.init(args=args)
    
    octo_bridge_node = OctoBridgeNode()
    rclpy.spin(octo_bridge_node)

    octo_bridge_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
