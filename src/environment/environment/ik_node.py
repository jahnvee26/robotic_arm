#!/usr/bin/env python3
"""
ROS2 Inverse Kinematics Node

This node subscribes to target positions and publishes corresponding joint angles
using inverse kinematics calculations.
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Pose
from sensor_msgs.msg import JointState
import math
from typing import List, Tuple, Optional, Union
import numpy as np
#from InverseKinematics import Kinematic  
from environment.InverseKinematics import Kinematic

# Try to import tf_transformations, fallback to manual conversion
try:
    import tf_transformations
    HAS_TF_TRANSFORMATIONS = True
except ImportError:
    HAS_TF_TRANSFORMATIONS = False

def quaternion_to_euler(quaternion):
    """
    Convert quaternion to Euler angles (roll, pitch, yaw)
    Args:
        quaternion: [x, y, z, w] format
    Returns:
        (roll, pitch, yaw) in radians
    """
    x, y, z, w = quaternion
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw

class IKNode(Node):
    """
    ROS2 Node for Inverse Kinematics calculations
    
    Subscribes to: /target_pose (geometry_msgs/Pose)
    Publishes to: /desired_joint_angles (sensor_msgs/JointState)
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

        self.target_position = np.zeros(3)  # Target position [x, y, z]
        self.target_orientation = np.zeros(3)  # Target orientation [roll, pitch, yaw]
        self.desired_joint_angles = np.zeros(5)  # 5 joints

        self.create_subscription(
            Pose,
            '/target_pose',
            self.target_pose_callback,
            1
        )
    def target_pose_callback(self, msg: Pose):
        """
        Callback for receiving target pose messages with position and orientation.
        """
        # Extract position
        self.target_position = np.array([msg.position.x, msg.position.y, msg.position.z])
        
        # Extract orientation from quaternion
        quaternion = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        
        if HAS_TF_TRANSFORMATIONS:
            roll, pitch, yaw = tf_transformations.euler_from_quaternion(quaternion)
        else:
            roll, pitch, yaw = quaternion_to_euler(quaternion)
            
        self.target_orientation = np.array([roll, pitch, yaw])
        
        self.get_logger().info(
            f'Received target pose: '
            f'pos=[{msg.position.x:.3f}, {msg.position.y:.3f}, {msg.position.z:.3f}], '
            f'orient=[roll={roll:.3f}, pitch={pitch:.3f}, yaw={yaw:.3f}] rad'
        )
        self.get_logger().info(f'Note: Using pitch={pitch:.3f} rad ({np.degrees(pitch):.1f}°) for wrist, ignoring roll/yaw')
        
        # Process the target pose and publish joint angles
        # Robot constraints: roll=0, yaw=0 (handled by base), pitch=variable (wrist)
        # IK solver returns: [θ0, θ1, θ2, θ3] + pitch should be applied to wrist
        base_joint_angles = self.ik_solver.inverse_kinematics(self.target_position)
        
        if base_joint_angles is not None:
            # Extract the desired pitch from target orientation
            target_pitch = self.target_orientation[1]  # pitch from [roll, pitch, yaw]
            
            # Check the shape of the IK solution
            if isinstance(base_joint_angles, np.ndarray):
                num_joints = len(base_joint_angles)
            else:
                num_joints = len(base_joint_angles) if hasattr(base_joint_angles, '__len__') else 1
            
            self.get_logger().info(f'IK solver returned {num_joints} joint angles')
            
            if num_joints == 5:
                # IK solver already returns 5 joints, use directly and add pitch to the last joint
                self.desired_joint_angles = np.array(base_joint_angles)
                self.desired_joint_angles[4] = target_pitch  # Override the last joint with target pitch
                
                self.get_logger().info(
                    f'IK solution (5 joints): θ0={np.degrees(base_joint_angles[0]):.1f}°, '
                    f'θ1={np.degrees(base_joint_angles[1]):.1f}°, '
                    f'θ2={np.degrees(base_joint_angles[2]):.1f}°, '
                    f'θ3={np.degrees(base_joint_angles[3]):.1f}°, '
                    f'θ4_original={np.degrees(base_joint_angles[4]):.1f}°, '
                    f'wrist_pitch_override={np.degrees(target_pitch):.1f}°'
                )
            elif num_joints == 4:
                # IK solver returns 4 joints, add pitch as 5th joint
                self.desired_joint_angles = np.zeros(5)
                self.desired_joint_angles[:4] = base_joint_angles  # Base IK solution
                self.desired_joint_angles[4] = target_pitch        # Apply target pitch to wrist
                
                self.get_logger().info(
                    f'IK solution (4+1 joints): θ0={np.degrees(base_joint_angles[0]):.1f}°, '
                    f'θ1={np.degrees(base_joint_angles[1]):.1f}°, '
                    f'θ2={np.degrees(base_joint_angles[2]):.1f}°, '
                    f'θ3={np.degrees(base_joint_angles[3]):.1f}°, '
                    f'wrist_pitch={np.degrees(target_pitch):.1f}°'
                )
            else:
                self.get_logger().error(f'Unexpected number of joints from IK solver: {num_joints}')
                return
        else:
            self.get_logger().warn('No valid IK solution found for target position')
            return

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
        self.get_logger().info(f'Joint mapping: [base_rot, shoulder, elbow, wrist_rot, wrist_pitch]')    

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