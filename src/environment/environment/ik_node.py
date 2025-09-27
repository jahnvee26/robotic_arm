#!/usr/bin/env python3
"""
ROS2 Inverse Kinematics Node

This node subscribes to target positions and publishes corresponding joint angles
using inverse kinematics calculations.
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from sensor_msgs.msg import JointState
import math
from typing import List, Tuple, Optional, Union
import numpy as np
# from InverseKinematics import Kinematic  # Assuming this is your IK logic module
from environment.InverseKinematics import Kinematic

class IKNode(Node):
    """
    ROS2 Node for Inverse Kinematics calculations
    
    Subscribes to: /target_position (geometry_msgs/Point)
    Publishes to: /target_joint_angles (sensor_msgs/JointState)
    """

    def __init__(self):
        super().__init__('ik_node')
        self.get_logger().info('IK Node Initialized')
        # Initialize the kinematic solver
        self.ik_solver = Kinematic()
        
        # Create publisher for joint angles
        self.joint_publisher = self.create_publisher(
            JointState,
            '/desired_joint_angles',
            1
        )

        self.target_position = np.zeros(3)  # Single position
        self.desired_joint_angles = np.zeros(5)  # 5 joints

        self.create_subscription(
            Point,
            '/target_position',
            self.target_position_callback,
            1
        )
    def target_position_callback(self, msg: Point):
        """
        Callback for receiving target position messages.
        """
        self.target_position = np.array([msg.x, msg.y, msg.z])
        self.get_logger().info(f'Received target position: [{msg.x:.3f}, {msg.y:.3f}, {msg.z:.3f}]')
        
        # Process the target position and publish joint angles
        self.desired_joint_angles = self.ik_solver.inverse_kinematics(self.target_position)

        self.publish_joint_angles()

    def publish_joint_angles(self):
        """
        Publish the calculated joint angles.
        """
        if self.desired_joint_angles is None:
            self.get_logger().warn('No valid IK solution found for the target position.')
            return
        
        joint_state_msg = JointState()
        joint_state_msg.header.stamp = self.get_clock().now().to_msg()
        
        # Handle both numpy array and list cases
        if isinstance(self.desired_joint_angles, np.ndarray):
            joint_state_msg.position = self.desired_joint_angles.tolist()
        else:
            joint_state_msg.position = list(self.desired_joint_angles)
        
        self.joint_publisher.publish(joint_state_msg)
        
        self.get_logger().info(f'Published joint angles (degrees): {np.degrees(self.desired_joint_angles)}')    

def main(args=None):
    rclpy.init(args=args)
    
    ik_node = IKNode()
    
    try:
        rclpy.spin(ik_node)
    except KeyboardInterrupt:
        pass
    finally:
        ik_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()