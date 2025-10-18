#!/usr/bin/env python3

"""
Octo Bridge Script (ROS2 Node)

This script acts as a bridge between ROS2 and the standalone Octo inference script.

Subscribes to: 
- /camera/image (sensor_msgs/Image)
- /text_prompt (std_msgs/String)

Publishes to: 
- /target_pose (geometry_msgs/Pose) - Full 6-DOF pose with position and orientation
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Point, Pose
from cv_bridge import CvBridge
import cv2
import numpy as np
import subprocess
import json
import tempfile
import os
import threading
import time
import math
import numpy as np

# Try to import tf_transformations, fallback to manual conversion
try:
    import tf_transformations
    HAS_TF_TRANSFORMATIONS = True
except ImportError:
    HAS_TF_TRANSFORMATIONS = False

def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert Euler angles to quaternion (w, x, y, z)
    """
    # Convert to radians if in degrees
    roll = math.radians(roll) if abs(roll) > 2*math.pi else roll
    pitch = math.radians(pitch) if abs(pitch) > 2*math.pi else pitch  
    yaw = math.radians(yaw) if abs(yaw) > 2*math.pi else yaw
    
    # Calculate quaternion components
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return [x, y, z, w]  # Return in ROS format [x, y, z, w]

def rpy_to_rotmat(roll, pitch, yaw):
    """
    Convert roll, pitch, yaw to rotation matrix
    """
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])
    
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])
    
    return Rz @ Ry @ Rx

def neutralize_pose_position(pose_data):
    """
    Apply inverse transformation to neutralize the pose orientation effects on position
    Returns the neutral position where the gripper center would be with zero orientation
    """
    # Extract position and orientation
    pos = np.array([pose_data.get('x', 0.0), pose_data.get('y', 0.0), pose_data.get('z', 0.0)])
    roll = pose_data.get('roll', 0.0)
    pitch = pose_data.get('pitch', 0.0)
    yaw = pose_data.get('yaw', 0.0)
    
    # Get rotation matrix
    R = rpy_to_rotmat(roll, pitch, yaw)
    
    # Apply inverse transformation to get neutral position
    # The idea is: if the gripper was rotated, where would its center be
    # if it was in neutral orientation?
    R_inv = R.T  # Inverse of rotation matrix is its transpose
    
    # Apply inverse rotation to position to get neutral coordinates
    neutral_pos = R_inv @ pos
    
    return {
        'x': float(neutral_pos[0]),
        'y': float(neutral_pos[1]),
        'z': float(neutral_pos[2]),
        'original_x': float(pos[0]),
        'original_y': float(pos[1]),
        'original_z': float(pos[2]),
        'original_roll': roll,
        'original_pitch': pitch,
        'original_yaw': yaw
    }

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
        self.current_top_view_image = None
        self.current_prompt = "pick up the cube"  # Default prompt
        self.processing_lock = threading.Lock()
        self.inference_completed = False  # Track if inference has been run
        
        # Action sequence execution state (similar to VLA node)
        self.action_sequence = None
        self.current_timestep = 0
        self.sequence_length = 0
        
        # Path to conda environment and Octo script
        self.conda_env = "octo"
        self.octo_script = "/home/welgpu/jahnvee/ddp/robotic_arm/src/environment/environment/octo_inference.py"
        
        # Performance configuration parameters
        self.declare_parameter('execution_rate', 2.0)     # Hz (2 Hz = every 0.5 seconds) 
        self.declare_parameter('use_image_caching', True) # Cache identical images
        
        execution_rate = self.get_parameter('execution_rate').value
        self.use_caching = self.get_parameter('use_image_caching').value
        
        self.get_logger().info(f"🔧 Performance config: execution={execution_rate}Hz, caching={self.use_caching}")
        
        # Publishers
        self.pose_publisher = self.create_publisher(
            Pose,
            '/target_pose',
            10
        )
        
        # Subscribers
        self.side_image_subscription = self.create_subscription(
            Image,
            '/camera/image',
            self.side_image_callback,
            10
        )
        
        self.top_image_subscription = self.create_subscription(
            Image,
            '/camera/top_view',
            self.top_image_callback,
            10
        )

        self.prompt_subscription = self.create_subscription(
            String,
            '/text_prompt',
            self.prompt_callback,
            10
        )
        
        # Processing timer (run inference once when image is available)
        self.inference_timer = self.create_timer(2.0, self.check_and_run_inference)  # Check every 2 seconds for first image
        
        # Action execution timer (execute actions from sequence at higher frequency)
        self.execution_timer = self.create_timer(1.0/execution_rate, self.execute_next_action)
        
        self.get_logger().info("🚀 Octo Bridge Node initialized - connecting ROS2 to Octo conda environment")

    def side_image_callback(self, msg: Image):
        """Callback for receiving side view camera images"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            with self.processing_lock:
                self.current_image = cv_image
                
                self.get_logger().debug(
                    f"Received side view image: {msg.width}x{msg.height}, "
                    f"frame_id: {msg.header.frame_id}"
                )
                
        except Exception as e:
            self.get_logger().error(f"Error processing side view image: {e}")

    def top_image_callback(self, msg: Image):
        """Callback for receiving top view camera images"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            with self.processing_lock:
                self.current_top_view_image = cv_image
                
                self.get_logger().debug(
                    f"Received top view image: {msg.width}x{msg.height}, "
                    f"frame_id: {msg.header.frame_id}"
                )
                
        except Exception as e:
            self.get_logger().error(f"Error processing top view image: {e}")
    def prompt_callback(self, msg: String):
        """Callback for receiving text prompts"""
        with self.processing_lock:
            self.current_prompt = msg.data
            self.get_logger().info(f"Received prompt: '{self.current_prompt}'")

    def check_and_run_inference(self):
        """Run inference only once when image is available"""
        if self.inference_completed:
            return  # Inference already completed
            
        if self.current_image is not None:
            self.get_logger().info(" Camera image available - running one-time Octo inference")
            self.run_inference()
            self.inference_completed = True
            # Cancel the timer since we only need to run once
            self.inference_timer.cancel()
            self.get_logger().info("✅ One-time inference completed - timer disabled")
        else:
            self.get_logger().debug(" Waiting for camera image...")

    def run_inference(self):
        """Run Octo inference once via subprocess to conda environment"""
        start_time = time.time()
        
        # Get current state
        with self.processing_lock:
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
                    self.get_logger().info(f" Generated {len(octo_result['target_poses'])} target poses")
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
            if 'target_poses' in octo_result and octo_result['target_poses']:
                self.action_sequence = octo_result['target_poses']
                self.sequence_length = len(self.action_sequence)
                self.current_timestep = 0
                
                self.get_logger().info(f"📦 Stored action sequence: {self.sequence_length} timesteps")
            else:
                self.get_logger().warn("No valid target poses in Octo result")

    def execute_next_action(self):
        """Execute the next action in the current sequence"""
        with self.processing_lock:
            # Check if we have a valid action sequence
            if (self.action_sequence is None or 
                self.current_timestep >= self.sequence_length):
                return  # No sequence or sequence completed
            
            # Get the current timestep action
            pose_data = self.action_sequence[self.current_timestep]
            
            # SIMPLIFIED APPROACH: Use original positions directly, ignore orientation effects
            # The complex transformation was giving wrong coordinates
            target_pose = Pose()
            
            # Use original position directly (no neutralization for now)
            target_pose.position.x = pose_data.get('x', 0.0)
            target_pose.position.y = pose_data.get('y', 0.0) 
            target_pose.position.z = pose_data.get('z', 0.0)
            
            # Orientation - all zeros (neutralized)
            # For this robot: roll=0, yaw=0, pitch=0 (no orientation)
            roll = 0.0   # Always zero for this robot
            pitch = 0.0  # ZERO (no orientation)
            yaw = 0.0    # Always zero for this robot
            
            # Convert Euler angles to quaternion (identity quaternion for zero rotation)
            if HAS_TF_TRANSFORMATIONS:
                quaternion = tf_transformations.quaternion_from_euler(roll, pitch, yaw)
            else:
                quaternion = euler_to_quaternion(roll, pitch, yaw)
                
            target_pose.orientation.x = quaternion[0]
            target_pose.orientation.y = quaternion[1]
            target_pose.orientation.z = quaternion[2]
            target_pose.orientation.w = quaternion[3]
            
            self.pose_publisher.publish(target_pose)
            
            # Log simplified approach
            self.get_logger().info(
                f" Executed timestep {self.current_timestep + 1}/{self.sequence_length}: "
                f"x={target_pose.position.x:.3f}, y={target_pose.position.y:.3f}, z={target_pose.position.z:.3f} "
                f"(using original positions, no transformation)"
            )
            
            # Move to next timestep
            self.current_timestep += 1
            
            # Check if sequence is complete
            if self.current_timestep >= self.sequence_length:
                self.get_logger().info("✅ Action sequence completed!")
                self.action_sequence = None  # Clear completed sequence

#Future use in case of manual re-triggering
    # def trigger_new_inference(self):
    #     """Manually trigger a new inference (useful for testing new prompts)"""
    #     with self.processing_lock:
    #         self.inference_completed = False
    #         self.action_sequence = None
    #         self.current_timestep = 0
    #     self.get_logger().info("🔄 Inference reset - will run on next check")
    #     # Restart the timer
    #     self.inference_timer = self.create_timer(2.0, self.check_and_run_inference)


def main(args=None):
    rclpy.init(args=args)
    
    octo_bridge_node = OctoBridgeNode()
    rclpy.spin(octo_bridge_node)

    octo_bridge_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
