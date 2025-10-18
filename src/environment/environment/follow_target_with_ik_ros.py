#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import Pose, Point, Quaternion
from std_msgs.msg import Header
import numpy as np
import cv2
from cv_bridge import CvBridge

# import sys
# sys.path.append("/home/welgpu/jahnvee/isaac-sim/isaac-sim-standalone-5.0.0-linux-x86_64/kit/python/lib/python3.11/site-packages")

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import carb
from isaacsim.core.api import World
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.examples.franka.tasks import FollowTarget
from isaacsim.robot.manipulators.examples.franka import KinematicsSolver
from isaacsim.core.utils.rotations import euler_angles_to_quat

import omni.kit.commands

class IsaacSimEnvironmentNode(Node):
    def __init__(self):
        super().__init__('isaac_sim_environment')
        
        # ROS2 Subscribers and Publishers
        self.delta_pose_subscription = self.create_subscription(
            Pose,
            '/delta_pose_commands',
            self.update_robot_pose,
            10
        )

        # Isaac Sim setup
        self.setup_isaac_sim()
        
        self.get_logger().info('Isaac Sim Environment Node started')
        self.initial_position = [0.5, 0.0, 0.5]
        self.initial_orientation = [0.0, 0.7071, 0.0, 0.7071]

        self.gripper_position = np.array(self.initial_position)
        self.gripper_orientation = np.array(self.initial_orientation)

        # World step timer
        self.create_timer(1/60, self.world_step)

    def setup_isaac_sim(self):
        self.my_world = World(stage_units_in_meters=1.0)
        my_task = FollowTarget(name="follow_target_task")
        self.my_world.add_task(my_task)
        self.my_world.reset()
        task_params = self.my_world.get_task("follow_target_task").get_params()
        self.franka_name = task_params["robot_name"]["value"]
        # target_name = task_params["target_name"]["value"]
        my_franka = self.my_world.scene.get_object(self.franka_name)
        self.my_controller = KinematicsSolver(my_franka)
        self.articulation_controller = my_franka.get_articulation_controller()
        self.reset_needed = False

    def world_step(self):
        if not simulation_app.is_running():
            return
            
        self.my_world.step(render=True)
        if self.my_world.is_stopped() and not self.reset_needed:
            self.reset_needed = True
        if self.my_world.is_playing():
            if self.reset_needed:
                self.my_world.reset()
                self.reset_needed = False
            actions, succ = self.my_controller.compute_inverse_kinematics(
                target_position=self.gripper_position,
                target_orientation=self.gripper_orientation,
            )
            if succ:
                # self.get_logger().info(f'Applying actions: {actions}')
                self.articulation_controller.apply_action(actions)
            else:
                self.get_logger().warning("IK did not converge to a solution.  No action is being taken.")

    def update_robot_pose(self, msg):
        """Set robot joint positions from ROS2 topic (5 real joints + 5 mimic joints)"""
        # Update the robot's gripper pose based on the incoming message
        self.get_logger().info(f'Received delta pose command: Position({msg.position.x}, {msg.position.y}, {msg.position.z}), Orientation({msg.orientation.x}, {msg.orientation.y}, {msg.orientation.z}, {msg.orientation.w})')
        delta_position = np.array([msg.position.x, msg.position.y, msg.position.z])
        delta_orientation = np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        self.gripper_position += delta_position
        self.gripper_orientation += delta_orientation
        return

def main(args=None):
    rclpy.init(args=args)

    isaac_sim_node = IsaacSimEnvironmentNode()
    rclpy.spin(isaac_sim_node)

    isaac_sim_node.destroy_node()

    rclpy.shutdown()
    # simulation_app.close()

if __name__ == '__main__':
    main()