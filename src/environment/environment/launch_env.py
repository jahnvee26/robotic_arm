#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import numpy as np

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
import omni.kit.commands
import omni.physx

import sys
import os
# sys.path.append("/home/welgpu/jahnvee/ddp/octo_main_ws/src")

class IsaacSimEnvironmentNode(Node):
    def __init__(self):
        super().__init__('isaac_sim_environment')
        
        # ROS2 Subscribers and Publishers
        self.joint_subscription = self.create_subscription(
            JointState,
            '/desired_joint_angles',
            self.joint_angles_callback,
            10
        )
        
        # Isaac Sim setup
        self.setup_isaac_sim()
        
        self.get_logger().info('Isaac Sim Environment Node started')

        # TODO: figure out what rate is good for this
        self.create_timer(0.01, self.world_step)

    def setup_isaac_sim(self):
        """Initialize Isaac Sim world and robot"""
        # Create world
        self.world = World(stage_units_in_meters=1.0)
        
        # Add ground & props
        asset_root = get_assets_root_path()
        self.world.scene.add_default_ground_plane()
        add_reference_to_stage(asset_root + "/Isaac/Samples/Props/Stage/Props_Table.usd", "/World/Table")

        # Import robot URDF
        roarm_urdf_path = "/home/welgpu/jahnvee/ddp/octo_main_ws/src/robots/roarm.urdf"
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
        mesh_pkg_path = "/home/welgpu/jahnvee/ddp/octo_main_ws/src/robots"
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
