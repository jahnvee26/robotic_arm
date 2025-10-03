#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import Header
import numpy as np
import cv2
from cv_bridge import CvBridge

# import sys
# sys.path.append("/home/welgpu/jahnvee/isaac-sim/isaac-sim-standalone-5.0.0-linux-x86_64/kit/python/lib/python3.11/site-packages")

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.api.robots import Robot
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.asset.importer.urdf import _urdf
from isaacsim.sensors.camera import Camera
import isaacsim.core.utils.numpy.rotations as rot_utils
import omni.kit.commands
import omni.physx

import sys
import os
# sys.path.append("/home/welgpu/jahnvee/ddp/octo_main_ws/src")

class IsaacSimEnvironmentNode(Node):
    def __init__(self):
        super().__init__('isaac_sim_environment')
        
        # Initialize CV Bridge for camera
        self.bridge = CvBridge()
        
        # ROS2 Subscribers and Publishers
        self.joint_subscription = self.create_subscription(
            JointState,
            '/desired_joint_angles',
            self.joint_angles_callback,
            10
        )
        # Camera publisher
        self.camera_publisher = self.create_publisher(
            Image,
            '/camera/image',
            10
        )
        # Isaac Sim setup
        self.setup_isaac_sim()
        
        self.get_logger().info('Isaac Sim Environment Node started with camera')

        # World step timer
        self.create_timer(0.01, self.world_step)
        
        # Camera publishing timer (10 Hz) - start with a small delay
        self.create_timer(1.0, self.start_camera_publishing)

    def setup_isaac_sim(self):
        """Initialize Isaac Sim world and robot"""
        # Create world
        self.world = World(stage_units_in_meters=1.0)
        
        # Add ground & props
        asset_root = get_assets_root_path()
        self.world.scene.add_default_ground_plane()
        add_reference_to_stage(asset_root + "/Isaac/Samples/Props/Stage/Props_Table.usd", "/World/Table")

        # Import robot URDF
        roarm_urdf_path = "/home/welgpu/jahnvee/ddp/robotic_arm/src/environment/resource/roarm.urdf"
        robot_usd_path = "/tmp/roarm_robot.usd"
        robot_prim_path = "/World/RoArm"

        # Configure URDF import
        import_config = _urdf.ImportConfig()
        import_config.fix_base = True
        import_config.merge_fixed_joints = False
        import_config.self_collision = False
        import_config.convex_decomp = False
        import_config.make_default_prim = True
        import_config.default_drive_strength = 1047.19751
        import_config.default_position_drive_damping = 52.35988

        # Set up mesh paths
        mesh_pkg_path = "/home/welgpu/jahnvee/ddp/octo_main_ws/src/environment/resource/meshes"
        os.environ["ROS_PACKAGE_PATH"] = mesh_pkg_path

        # Import URDF
        ok, art_root_path = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=roarm_urdf_path,
            import_config=import_config,
            dest_path=robot_usd_path,
            get_articulation_root=True,
        )
        if not ok:
            raise RuntimeError("URDF import failed")
        
        add_reference_to_stage(robot_usd_path, robot_prim_path)
        art_root_path = robot_prim_path
        self.get_logger().info(f"URDF imported successfully. Articulation root: {art_root_path}")


        # Step a few frames for visuals to settle
        for _ in range(5):
            self.world.step(render=True)

        # Create robot object
        self.robot = Robot(
            prim_path=art_root_path,
            name="RoArm",
            position=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([0.0, 0.0, 0.0, 1.0])
        )

        self.world.scene.add(self.robot)
        self.get_logger().info("Robot added to scene successfully")

        # Reset and initialize
        self.world.reset()
        import time
        time.sleep(1.0)
        
        if self.robot:
            self.robot.initialize()
            self.get_logger().info("Robot initialized successfully")

            # Joint names
            self.joint_names = list(self.robot.dof_names)
            self.get_logger().info(f"Robot DOF names: {self.joint_names}")
            

        self.get_logger().info("Isaac Sim environment setup complete")

        position = np.array([0.2, 0.1, 0.05])  
        #print(f"[CUBE] Creating cube at position: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]")
        cube = self.world.scene.add(
            DynamicCuboid(
                name="cube",
                position=np.array([position[0], position[1], position[2]]),
                prim_path="/World/Cube",
                scale=np.array([0.05, 0.05, 0.05]),  # Very small for precise gripping
                size=1.0,
                color=np.array([0, 0, 1])
            )
        )
        
        # Setup camera for VLA
        self.setup_camera()

    def setup_camera(self):
        """Setup camera for VLA vision input"""
        try:
            self.get_logger().info(" Setting up camera...")
            
            camera_position = np.array([0.9, -1.2, 0.5])  # Back and to the side, elevated
        
            # Calculate orientation to look toward the robot workspace center
            target_position = np.array([0.1, 0.05, 0.2])  # Center between robot and cube

            direction = target_position - camera_position
            direction = direction / np.linalg.norm(direction)
            
            # Calculate pitch (up/down rotation)
            pitch = -np.arcsin(direction[2])  # Negative because we want to look down
            
            # Calculate yaw (left/right rotation)  
            yaw = np.arctan2(direction[1], direction[0])
            
            # Convert to degrees and create orientation
            euler_angles = np.array([0, np.degrees(pitch), np.degrees(yaw)])
            camera_orientation = rot_utils.euler_angles_to_quats(euler_angles, degrees=True)
            
            self.camera = Camera(
                prim_path="/World/Camera",
                position=camera_position,
                frequency=20,
                resolution=(256, 256),
                orientation=camera_orientation
            )
            
            self.get_logger().info("Camera created, adding to world...")
            
            # Add camera to world and initialize
            self.world.scene.add(self.camera)
            self.camera.initialize()
            
            self.get_logger().info("✅ Camera setup complete - positioned to view robot and cube")
            
        except Exception as e:
            self.get_logger().error(f"❌ Failed to setup camera: {e}")
            self.camera = None
    
    def start_camera_publishing(self):
        """Start camera publishing after Isaac Sim is fully initialized"""
        #self.get_logger().info("Starting camera publishing...")
        # Cancel the one-time timer and start the regular camera publishing
        self.camera_publishing_timer = self.create_timer(0.1, self.publish_camera_image)
    
    def publish_camera_image(self):
        """Capture and publish camera image for VLA"""
        try:
            if not hasattr(self, 'camera') or self.camera is None:
                return
            
            # Get current frame first
            self.camera.get_current_frame()
            
            # Get RGB image
            rgb_data = self.camera.get_rgba()
            if rgb_data is None:
                self.get_logger().warn("No camera data available")
                return
                
            # Convert from RGBA to RGB
            rgb_image = rgb_data[:, :, :3]
            
            # Convert from float [0,1] to uint8 [0,255]
            if rgb_image.dtype == np.float32 or rgb_image.dtype == np.float64:
                rgb_image = (rgb_image * 255).astype(np.uint8)
            
            # Convert RGB to BGR for ROS
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            
            # Create ROS Image message
            ros_image = self.bridge.cv2_to_imgmsg(bgr_image, "bgr8")
            ros_image.header = Header()
            ros_image.header.stamp = self.get_clock().now().to_msg()
            ros_image.header.frame_id = "isaac_sim_camera"
            
            # Publish image
            self.camera_publisher.publish(ros_image)
            
        except Exception as e:
            self.get_logger().error(f"❌ Camera publishing error: {e}")

    def world_step(self):
        if simulation_app.is_running():
            self.world.step(render=True)   
            
    def convert_to_sim_angles(self, joint_angles):

        sim_angles = np.zeros_like(joint_angles)
        sim_angles[0] = + joint_angles[0] - np.pi/2
        sim_angles[1] = - joint_angles[1]
        sim_angles[2] = - joint_angles[2]
        sim_angles[3] = - joint_angles[3]
        sim_angles[4] =   joint_angles[4]

        return sim_angles

    def add_mimic_joints(self):
        """
        Create full 10-joint array with mimic joints from 5 real joint angles
        """
        full_joint_angles = np.zeros(10, dtype=float)
        full_joint_angles[:5] = self.joint_angles
        full_joint_angles[5] = -self.joint_angles[4]      # L4_to_L5_1_B
        full_joint_angles[6] = self.joint_angles[4]       # L4_to_L5_2_A
        full_joint_angles[7] = -self.joint_angles[4]      # L4_to_L5_2_B
        full_joint_angles[8] = -self.joint_angles[4]      # L5_1_A_to_L5_3_A
        full_joint_angles[9] = self.joint_angles[4]       # L5_1_B_to_L5_3_B (mimics joint 5)
        return full_joint_angles
    
    def send_joint_angles(self):
        # Create full 10-joint array with mimic joints
        full_joint_angles_array = self.add_mimic_joints()

        # Verify we have the right number of joints for the robot
        if len(full_joint_angles_array) != len(self.joint_names):
            self.get_logger().warning(f"Expected {len(self.joint_names)} total joints to be sent for action, got {len(full_joint_angles_array)}")
            return

        action = ArticulationAction(joint_positions=full_joint_angles_array.tolist())
        self.robot.apply_action(action)
        self.world.step(render=True)
        self.get_logger().info(f"Sent action to sim")

    def joint_angles_callback(self, msg):
        """Set robot joint positions from ROS2 topic (5 real joints + 5 mimic joints)"""
        self.get_logger().info(f"Received joint angles message: {np.degrees(msg.position)}")
        self.joint_angles = self.convert_to_sim_angles(np.array(msg.position))

        # Expect only 5 real joint angles in degrees
        if len(self.joint_angles) != 5:
            self.get_logger().warning(f"Expected 5 joint angles, got {len(self.joint_angles)}")
            return
        self.send_joint_angles()



def main(args=None):
    rclpy.init(args=args)

    isaac_sim_node = IsaacSimEnvironmentNode()
    rclpy.spin(isaac_sim_node)

    isaac_sim_node.destroy_node()

    rclpy.shutdown()
    # simulation_app.close()

if __name__ == '__main__':
    main()