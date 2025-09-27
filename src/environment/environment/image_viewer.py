#!/usr/bin/env python3

"""
Camera Image Viewer

This node subscribes to camera images and displays them in a CV2 window for verification.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import numpy as np
from cv_bridge import CvBridge

class ImageViewer(Node):
    """Simple image viewer for camera feed verification"""

    def __init__(self):
        super().__init__('image_viewer')
        
        self.bridge = CvBridge()
        
        # Subscribe to camera images
        self.image_subscription = self.create_subscription(
            Image,
            '/camera/image',
            self.image_callback,
            10
        )
        
        # Create OpenCV window
        cv2.namedWindow("Isaac Sim Camera Feed", cv2.WINDOW_AUTOSIZE)
        
        self.get_logger().info("Image Viewer started - Press 'q' to quit")
        self.frame_count = 0

    def image_callback(self, msg: Image):
        """Display received images"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Add frame info overlay
            self.frame_count += 1
            info_text = f"Frame: {self.frame_count}, Size: {msg.width}x{msg.height}"
            info_text += f", Frame ID: {msg.header.frame_id}"
            
            # Add text overlay
            cv2.putText(cv_image, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Add timestamp
            timestamp = f"Time: {msg.header.stamp.sec}.{msg.header.stamp.nanosec//1000000:03d}"
            cv2.putText(cv_image, timestamp, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Show image
            cv2.imshow("Isaac Sim Camera Feed", cv_image)
            
            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.get_logger().info("Quit requested")
                rclpy.shutdown()
                
            # Log reception
            if self.frame_count % 30 == 0:  # Every 30 frames (3 seconds at 10Hz)
                mean_brightness = np.mean(cv_image)
                self.get_logger().info(f"Received {self.frame_count} frames, brightness: {mean_brightness:.1f}")
                
        except Exception as e:
            self.get_logger().error(f"Error displaying image: {e}")

    def destroy_node(self):
        """Clean up OpenCV windows"""
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        viewer = ImageViewer()
        rclpy.spin(viewer)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            viewer.destroy_node()
        except:
            pass
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
