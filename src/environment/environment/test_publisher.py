#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Pose
from std_msgs.msg import Header
import time

class TestPublisherNode(Node):
    def __init__(self):
        super().__init__('test_publisher')
        
        # Publishers for testing the IK system
        self.position_publisher = self.create_publisher(Point, 'target_position', 10)
        self.pose_publisher = self.create_publisher(Pose, 'target_pose', 10)
        
        # Timer to publish test targets
        self.timer = self.create_timer(5.0, self.publish_test_targets)
        self.target_index = 0
        
        # Define test targets
        self.test_positions = [
            [0.25, 0.1, 0.45],   # Cube position
            [0.20, 0.0, 0.30],   # Closer position
            [0.30, 0.15, 0.40],  # Different angle
            [0.15, -0.1, 0.35],  # Left side
        ]
        
        self.get_logger().info('Test Publisher Node started - will publish targets every 5 seconds')

    def publish_test_targets(self):
        """Publish test target positions cyclically"""
        if self.target_index < len(self.test_positions):
            pos = self.test_positions[self.target_index]
            
            # Publish as Point message
            point_msg = Point()
            point_msg.x = pos[0]
            point_msg.y = pos[1] 
            point_msg.z = pos[2]
            
            self.position_publisher.publish(point_msg)
            
            self.get_logger().info(f"Published target {self.target_index + 1}: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
            
            self.target_index = (self.target_index + 1) % len(self.test_positions)

def main(args=None):
    rclpy.init(args=args)
    
    try:
        test_node = TestPublisherNode()
        rclpy.spin(test_node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            test_node.destroy_node()
        except:
            pass
        rclpy.shutdown()

if __name__ == '__main__':
    main()
