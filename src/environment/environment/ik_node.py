#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Point, Quaternion
from std_msgs.msg import Float64MultiArray
import numpy as np

# import sys
# sys.path.append("/home/welgpu/jahnvee/ddp/octo_main_ws/src")

from roarm_pick_place_controller import IK5DOF, forward_kinematics

class InverseKinematicsNode(Node):
    def __init__(self):
        super().__init__('inverse_kinematics_node')
        
        # Initialize IK solver
        self.ik_solver = IK5DOF()
        
        # ROS2 Subscribers and Publishers
        self.target_pose_subscription = self.create_subscription(
            Pose,
            'target_pose',
            self.target_pose_callback,
            10
        )
        
        self.target_position_subscription = self.create_subscription(
            Point,
            'target_position',
            self.target_position_callback,
            10
        )
        
        self.joint_angles_publisher = self.create_publisher(
            JointState,
            'target_joint_angles',
            10
        )
        
        # Service for one-shot IK calculation
        # self.ik_service = self.create_service(
        #     # You can add a custom service type for IK requests if needed
        # )
        
        self.get_logger().info('Inverse Kinematics Node started')
        self.get_logger().info('Subscribed to: target_pose, target_position')
        self.get_logger().info('Publishing to: target_joint_angles')

    def target_pose_callback(self, msg):
        """Callback for full 6DOF pose (position + orientation)"""
        try:
            # Extract position
            x = msg.position.x
            y = msg.position.y
            z = msg.position.z
            
            # Extract orientation (convert quaternion to Euler angles)
            qx, qy, qz, qw = msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
            
            # Convert quaternion to roll, pitch, yaw
            roll, pitch, yaw = self.quaternion_to_euler(qw, qx, qy, qz)
            
            self.get_logger().info(f"Received target pose: pos=({x:.3f}, {y:.3f}, {z:.3f}), "
                                 f"rpy=({np.degrees(roll):.1f}°, {np.degrees(pitch):.1f}°, {np.degrees(yaw):.1f}°)")
            
            # Solve IK (using pitch and roll, ignoring yaw for 5DOF arm)
            self.solve_and_publish_ik([x, y, z], pitch=pitch, roll=roll)
            
        except Exception as e:
            self.get_logger().error(f"Error in target pose callback: {e}")

    def target_position_callback(self, msg):
        """Callback for position-only target (3DOF)"""
        try:
            x = msg.x
            y = msg.y
            z = msg.z
            
            self.get_logger().info(f"Received target position: ({x:.3f}, {y:.3f}, {z:.3f})")
            
            # Solve IK with default orientation
            self.solve_and_publish_ik([x, y, z], pitch=0.0, roll=0.0)
            
        except Exception as e:
            self.get_logger().error(f"Error in target position callback: {e}")

    def solve_and_publish_ik(self, position, pitch=0.0, roll=0.0):
        """Solve inverse kinematics and publish joint angles"""
        try:
            x, y, z = position
            
            # Solve IK for 5DOF
            joint_angles_5 = self.ik_solver.inverse_kinematics([x, y, z], pitch=pitch, roll=roll)
            
            if joint_angles_5 is None:
                self.get_logger().warn(f"IK solution not found for target: ({x:.3f}, {y:.3f}, {z:.3f})")
                return
            
            self.get_logger().info(f"IK solution found: {np.degrees(joint_angles_5)}")
            
            # Verify with forward kinematics
            fk_matrix = forward_kinematics(joint_angles_5)
            fk_pos = fk_matrix[:3, 3]
            error = np.linalg.norm(fk_pos - np.array([x, y, z]))
            self.get_logger().info(f"FK verification - Target: ({x:.3f}, {y:.3f}, {z:.3f}), "
                                 f"FK result: ({fk_pos[0]:.3f}, {fk_pos[1]:.3f}, {fk_pos[2]:.3f}), "
                                 f"Error: {error:.4f}m")
            
            # Build full joint array with mimic joints (10 DOF total)
            joint_angles_full = self.build_full_joint_array(joint_angles_5)
            
            # Create and publish JointState message
            joint_state_msg = JointState()
            joint_state_msg.header.stamp = self.get_clock().now().to_msg()
            joint_state_msg.name = ['base_to_L1', 'L1_to_L2', 'L2_to_L3', 'L3_to_L4', 'L4_to_L5_1_A', 
                                  'L4_to_L5_1_B', 'L4_to_L5_2_A', 'L4_to_L5_2_B', 'L5_1_A_to_L5_3_A', 'L5_1_B_to_L5_3_B']
            joint_state_msg.position = joint_angles_full.tolist()
            joint_state_msg.velocity = [0.0] * 10  # Zero velocities
            
            self.joint_angles_publisher.publish(joint_state_msg)
            self.get_logger().info("Published joint angles to Isaac Sim")
            
        except Exception as e:
            self.get_logger().error(f"Error solving IK: {e}")

    def build_full_joint_array(self, joint_angles_5):
        """Build full 10-DOF joint array including mimic joints"""
        joint_angles = np.zeros(10, dtype=float)
        joint_angles[:5] = joint_angles_5
        
        # Mimic joint relationships from URDF
        joint_angles[5] = -joint_angles_5[4]      # L4_to_L5_1_B (mimic L4_to_L5_1_A with multiplier -1)
        joint_angles[6] = joint_angles_5[4]       # L4_to_L5_2_A (mimic L4_to_L5_1_A with multiplier 1)
        joint_angles[7] = -joint_angles_5[4]      # L4_to_L5_2_B (mimic L4_to_L5_1_A with multiplier -1)
        joint_angles[8] = -joint_angles_5[4]      # L5_1_A_to_L5_3_A (mimic L4_to_L5_1_A with multiplier -1)
        joint_angles[9] = joint_angles_5[4]       # L5_1_B_to_L5_3_B (mimic L4_to_L5_1_B with multiplier -1, which is joint_angles_5[4])
        
        return joint_angles

    def quaternion_to_euler(self, w, x, y, z):
        """Convert quaternion to Euler angles (roll, pitch, yaw)"""
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

def main(args=None):
    rclpy.init(args=args)
    
    try:
        ik_node = InverseKinematicsNode()
        
        # Example: Publish a test target after initialization
        import time
        time.sleep(2.0)  # Wait for node to initialize
        
        # Publish test target to cube position
        test_target = Point()
        test_target.x = 0.25
        test_target.y = 0.1
        test_target.z = 0.45
        
        ik_node.get_logger().info(f"Publishing test target: ({test_target.x}, {test_target.y}, {test_target.z})")
        
        # Create a timer to publish the test target once
        def publish_test_target():
            msg = Point()
            msg.x = 0.25
            msg.y = 0.1
            msg.z = 0.45
            ik_node.target_position_callback(msg)
        
        ik_node.create_timer(3.0, publish_test_target)  # Publish test target after 3 seconds
        
        rclpy.spin(ik_node)
        
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        try:
            ik_node.destroy_node()
        except:
            pass
        rclpy.shutdown()

if __name__ == '__main__':
    main()
