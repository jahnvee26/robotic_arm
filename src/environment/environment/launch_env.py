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
            'target_joint_angles',
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
    def world_step(self):
        if simulation_app.is_running():
            self.world.step(render=True)

    def joint_angles_callback(self, msg):
        """Directly set robot joint positions from ROS2 topi"""
        target_angles = np.array(msg.position)
        if len(target_angles) != len(self.joint_names):
            self.get_logger().warn(
                f"Expected {len(self.joint_names)} joints, got {len(target_angles)}"
            )
            return

        action = ArticulationAction(joint_positions=target_angles.tolist())
        self.robot.apply_action(action)
        self.world.step(render=True)

        self.get_logger().info(f"Applied joint angles: {np.degrees(target_angles[:5])}")
        
def main(args=None):
    rclpy.init(args=args)

    isaac_sim_node = IsaacSimEnvironmentNode()
    rclpy.spin(isaac_sim_node)

    isaac_sim_node.destroy_node()

    rclpy.shutdown()
    # simulation_app.close()

if __name__ == '__main__':
    main()
