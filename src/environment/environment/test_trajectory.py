#!/usr/bin/env python3
"""
Test Trajectory Publisher

This script publishes hardcoded target poses that mimic the VLA model output
to test if the robot arm moves correctly through the IK pipeline.

Publishes to: /target_pose (geometry_msgs/Pose)
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Pose
import time
import json
import os
import math

class TestTrajectoryNode(Node):
    """
    ROS2 Node that publishes target positions from JSON file for testing
    """

    def __init__(self):
        super().__init__('test_trajectory_node')
        
        # Publisher for target poses
        self.pose_publisher = self.create_publisher(
            Pose,
            '/target_pose',
            10
        )
        
        # Load test trajectory from JSON file
        self.test_trajectory = self.load_trajectory_from_json()
        
        # Execution state
        self.current_step = 0
        self.max_steps = len(self.test_trajectory)
        self.execution_rate = 2.0  # Hz (every 0.5 seconds)
        
        # Timer for publishing positions
        self.execution_timer = self.create_timer(
            1.0 / self.execution_rate, 
            self.publish_next_position
        )
        
        self.get_logger().info("🚀 Test Trajectory Node initialized")
        self.get_logger().info(f"Loaded {self.max_steps} test poses from JSON")
        self.print_trajectory_info()
    
    def load_trajectory_from_json(self):
        """Load trajectory data from JSON file"""
        # Use absolute path to the source directory
        json_file_path = '/home/welgpu/jahnvee/ddp/robotic_arm/src/environment/environment/test_trajectory.json'
        
        try:
            with open(json_file_path, 'r') as f:
                trajectory_data = json.load(f)
            
            # Extract target_positions from JSON
            if 'target_positions' in trajectory_data:
                self.get_logger().info(f"✅ Successfully loaded trajectory from: {json_file_path}")
                return trajectory_data['target_positions']
            else:
                self.get_logger().error("❌ No 'target_positions' found in JSON file")
                raise ValueError("Missing target_positions in JSON")
                
        except FileNotFoundError:
            self.get_logger().error(f"❌ JSON file not found: {json_file_path}")
            self.get_logger().error("❌ Please ensure test_trajectory.json exists in the source directory")
            raise FileNotFoundError(f"Cannot find {json_file_path}")
        except json.JSONDecodeError as e:
            self.get_logger().error(f"❌ Invalid JSON format: {e}")
            raise json.JSONDecodeError(f"JSON parsing failed: {e}")
    
    def print_trajectory_info(self):
        """Print the test trajectory information"""
        self.get_logger().info("🎯 Test Trajectory Plan:")
        for i, pos in enumerate(self.test_trajectory):
            # Check if orientation data exists
            if 'roll' in pos and 'pitch' in pos and 'yaw' in pos:
                self.get_logger().info(
                    f"  Step {i+1}: x={pos['x']:.3f}, y={pos['y']:.3f}, z={pos['z']:.3f}, "
                    f"roll={pos['roll']:.3f}, pitch={pos['pitch']:.3f}, yaw={pos['yaw']:.3f}"
                )
            else:
                self.get_logger().info(
                    f"  Step {i+1}: x={pos['x']:.3f}, y={pos['y']:.3f}, z={pos['z']:.3f} (position only)"
                )
        self.get_logger().info(f"⏱️ Execution rate: {self.execution_rate} Hz")
        self.get_logger().info("🚀 Starting execution in 2 seconds...")

    def publish_next_position(self):
        """Publish the next pose in the trajectory"""
        if self.current_step >= self.max_steps:
            self.get_logger().info("✅ Test trajectory completed!")
            self.execution_timer.cancel()
            return
        
        # Get current position
        pos_data = self.test_trajectory[self.current_step]
        
        # Create and publish target pose
        target_pose = Pose()
        
        # Set position
        target_pose.position.x = pos_data['x']
        target_pose.position.y = pos_data['y']
        target_pose.position.z = pos_data['z']
        
        # Set orientation (identity quaternion since roll=pitch=yaw=0)
        # Identity quaternion represents no rotation: [x=0, y=0, z=0, w=1]
        target_pose.orientation.x = 0.0
        target_pose.orientation.y = 0.0
        target_pose.orientation.z = 0.0
        target_pose.orientation.w = 1.0
        
        self.pose_publisher.publish(target_pose)
        
        # Log with simplified pose information (all orientations are zero)
        self.get_logger().info(
            f"📤 Published step {self.current_step + 1}/{self.max_steps}: "
            f"x={target_pose.position.x:.3f}, y={target_pose.position.y:.3f}, z={target_pose.position.z:.3f}, "
            f"roll=0.0, pitch=0.0, yaw=0.0"
        )
        
        # Move to next step
        self.current_step += 1

    def reset_trajectory(self):
        """Reset trajectory to start from beginning"""
        self.current_step = 0
        self.get_logger().info("🔄 Trajectory reset to beginning")
        
        # Restart timer if it was cancelled
        if self.execution_timer.is_canceled():
            self.execution_timer = self.create_timer(
                1.0 / self.execution_rate, 
                self.publish_next_position
            )


def main(args=None):
    rclpy.init(args=args)
    
    test_trajectory_node = TestTrajectoryNode()
    
    try:
        rclpy.spin(test_trajectory_node)
    except KeyboardInterrupt:
        test_trajectory_node.get_logger().info("🛑 Test trajectory stopped by user")
    
    test_trajectory_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()