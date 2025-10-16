#!/usr/bin/env python3
"""
Top View Camera Test

Creates an actual Isaac Sim top camera for testing different positions
without affecting the main launch_env.py
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
import numpy as np
import cv2
from cv_bridge import CvBridge

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.sensors.camera import Camera
import isaacsim.core.utils.numpy.rotations as rot_utils

class TopCameraTestNode(Node):
    def __init__(self):
        super().__init__('top_camera_test')
        
        self.bridge = CvBridge()
        
        # Publisher for test top camera
        self.top_camera_publisher = self.create_publisher(
            Image,
            '/camera/top_view_test',
            10
        )
        
        # Setup Isaac Sim world first, then camera
        self.setup_test_environment()
        
        self.get_logger().info('🎥 Top Camera Test with Isaac Sim started')

        # World step timer
        self.create_timer(0.01, self.world_step)
        
        # Setup camera after world is ready
        self.create_timer(2.0, self.setup_camera_delayed)
        
        # Camera publishing timer - start after camera setup
        self.camera_ready = False

    def setup_test_environment(self):
        """Setup test environment with objects but no camera yet"""
        # Create world
        self.world = World(stage_units_in_meters=1.0)
        
        # Add ground
        self.world.scene.add_default_ground_plane()
        
        # Add table for reference
        asset_root = get_assets_root_path()
        add_reference_to_stage(asset_root + "/Isaac/Samples/Props/Stage/Props_Table.usd", "/World/TestTable")

        # Add cube at same position as main environment
        cube_position = np.array([0.2, 0.1, 0.05])
        cube = self.world.scene.add(
            DynamicCuboid(
                name="test_cube",
                position=cube_position,
                prim_path="/World/TestCube",
                scale=np.array([0.05, 0.05, 0.05]),
                size=1.0,
                color=np.array([0, 0, 1])  # Blue like in main env
            )
        )
        
        self.get_logger().info("🌍 Test environment setup complete")

    def setup_camera_delayed(self):
        """Setup camera after world is fully initialized"""
        self.get_logger().info("🎥 Setting up camera after world initialization...")
        self.setup_top_camera()
        
        # Start publishing after camera setup
        self.create_timer(1.0, self.start_camera_publishing)

    def setup_top_camera(self):
        """Setup top camera - ADJUST THESE VALUES TO TEST"""
        self.get_logger().info("🎥 Setting up TOP CAMERA for testing...")
        
        # ========== ADJUST THESE VALUES FOR TESTING ==========
        top_camera_position = np.array([0.5, 0.0, 0.5])    # Camera position
        target_position = np.array([0.2, 0.1, 0.05])       # Look at cube directly
        # =====================================================
        
        # Calculate direction vector
        direction = target_position - top_camera_position
        direction = direction / np.linalg.norm(direction)
        
        # Calculate pitch and yaw
        pitch = -np.arcsin(direction[2])
        yaw = np.arctan2(direction[1], direction[0])
        
        # Convert to quaternion
        euler_angles = np.array([0, np.degrees(pitch), np.degrees(yaw)])
        camera_orientation = rot_utils.euler_angles_to_quats(euler_angles, degrees=True)
        
        # Generate unique camera name with timestamp
        import time
        unique_id = int(time.time() * 1000) % 10000  # Last 4 digits of timestamp
        camera_name = f"test_top_camera_{unique_id}"
        prim_path = f"/World/TestTopCamera_{unique_id}"
        
        try:
            self.top_camera = Camera(
                prim_path=prim_path,
                name=camera_name,
                position=top_camera_position,
                frequency=20,
                resolution=(256, 256),
                orientation=camera_orientation
            )
            
            # Add to world and initialize
            self.world.scene.add(self.top_camera)
            self.top_camera.initialize()
            
            self.get_logger().info(f"✅ TOP CAMERA TEST setup:")
            self.get_logger().info(f"   📛 Name: {camera_name}")
            self.get_logger().info(f"   📍 Position: [{top_camera_position[0]:.2f}, {top_camera_position[1]:.2f}, {top_camera_position[2]:.2f}]")
            self.get_logger().info(f"   🎯 Target: [{target_position[0]:.2f}, {target_position[1]:.2f}, {target_position[2]:.2f}]")
            self.get_logger().info(f"   📐 Pitch: {np.degrees(pitch):.1f}°, Yaw: {np.degrees(yaw):.1f}°")
            self.get_logger().info(f"   📡 Publishing to: /camera/top_view_test")
            
        except Exception as e:
            self.get_logger().error(f"❌ Failed to create camera: {e}")
            self.top_camera = None

    def start_camera_publishing(self):
        """Start camera publishing"""
        if not hasattr(self, 'top_camera') or self.top_camera is None:
            self.get_logger().error("❌ Camera not ready for publishing")
            return
            
        self.get_logger().info("📹 Starting top camera publishing...")
        
        # Test camera first
        try:
            self.top_camera.get_current_frame()
            rgb_data = self.top_camera.get_rgba()
            if rgb_data is not None:
                self.get_logger().info(f"✅ Camera test successful - data shape: {rgb_data.shape}")
                self.camera_ready = True
            else:
                self.get_logger().warning("⚠️ Camera returning None data")
        except Exception as e:
            self.get_logger().error(f"❌ Camera test failed: {e}")
            
        if self.camera_ready:
            self.camera_publishing_timer = self.create_timer(0.1, self.publish_top_camera)
        else:
            # Retry after a delay
            self.create_timer(1.0, self.start_camera_publishing)

    def publish_top_camera(self):
        """Publish top camera image"""
        if not self.camera_ready:
            return
            
        try:
            if not hasattr(self, 'top_camera') or self.top_camera is None:
                return
            
            # Get current frame
            self.top_camera.get_current_frame()
            
            # Get RGB image
            rgb_data = self.top_camera.get_rgba()
            if rgb_data is None:
                self.get_logger().debug("No RGBA data from camera")
                return
            
            # Debug: Check data shape
            if rgb_data.shape[0] == 0:
                self.get_logger().debug("Empty image data")
                return
            
            # Handle different data formats
            if len(rgb_data.shape) == 1:
                # Data is flattened, reshape it
                # Assuming 256x256 resolution as set in camera config
                try:
                    rgb_data = rgb_data.reshape(256, 256, 4)  # RGBA format
                except:
                    self.get_logger().error(f"Cannot reshape data with shape {rgb_data.shape}")
                    return
            
            if len(rgb_data.shape) != 3:
                self.get_logger().error(f"Unexpected image shape: {rgb_data.shape}")
                return
                
            # Convert from RGBA to RGB
            if rgb_data.shape[2] >= 3:
                rgb_image = rgb_data[:, :, :3]
            else:
                self.get_logger().error(f"Not enough channels in image: {rgb_data.shape[2]}")
                return
            
            # Convert to uint8
            if rgb_image.dtype == np.float32 or rgb_image.dtype == np.float64:
                # Clamp values to [0, 1] range first
                rgb_image = np.clip(rgb_image, 0.0, 1.0)
                rgb_image = (rgb_image * 255).astype(np.uint8)
            elif rgb_image.dtype != np.uint8:
                rgb_image = rgb_image.astype(np.uint8)
            
            # Convert RGB to BGR for ROS
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            
            # Create ROS Image message
            ros_image = self.bridge.cv2_to_imgmsg(bgr_image, "bgr8")
            ros_image.header = Header()
            ros_image.header.stamp = self.get_clock().now().to_msg()
            ros_image.header.frame_id = "test_top_camera"
            
            # Publish image
            self.top_camera_publisher.publish(ros_image)
            
        except Exception as e:
            self.get_logger().error(f"❌ Top camera publishing error: {e}")

    def world_step(self):
        """Step the simulation"""
        if simulation_app.is_running():
            self.world.step(render=True)

    def destroy_node(self):
        """Clean up resources when node is destroyed"""
        try:
            if hasattr(self, 'top_camera') and self.top_camera is not None:
                self.get_logger().info("🧹 Cleaning up camera...")
                # The world cleanup will handle camera cleanup
            if hasattr(self, 'world') and self.world is not None:
                self.get_logger().info("🧹 Cleaning up world...")
                self.world.clear()
        except Exception as e:
            self.get_logger().warning(f"⚠️ Cleanup warning: {e}")
        finally:
            super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        top_camera_test_node = TopCameraTestNode()
        rclpy.spin(top_camera_test_node)
    except KeyboardInterrupt:
        top_camera_test_node.get_logger().info("🛑 Top camera test stopped by user")
    
    top_camera_test_node.destroy_node()
    rclpy.shutdown()
    simulation_app.close()


if __name__ == '__main__':
    main()
