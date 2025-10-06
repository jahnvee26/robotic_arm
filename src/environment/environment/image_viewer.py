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
    """Simple image viewer for dual camera feed verification"""

    def __init__(self):
        super().__init__('image_viewer')
        
        self.bridge = CvBridge()
        
        # State for both camera feeds
        self.side_image = None
        self.top_image = None
        
        # Subscribe to both camera images
        self.side_image_subscription = self.create_subscription(
            Image,
            '/camera/image',
            self.side_image_callback,
            10
        )
        
        self.top_image_subscription = self.create_subscription(
            Image,
            '/camera/top_view',
            self.top_image_callback,
            10
        )
        
        # Create OpenCV windows
        cv2.namedWindow("Side View Camera", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("Top View Camera", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("Combined View", cv2.WINDOW_AUTOSIZE)
        
        # Position windows so they don't overlap
        cv2.moveWindow("Side View Camera", 100, 100)
        cv2.moveWindow("Top View Camera", 400, 100)
        cv2.moveWindow("Combined View", 100, 400)
        
        self.get_logger().info("🎥 Dual Image Viewer started - Press 'q' to quit")
        self.side_frame_count = 0
        self.top_frame_count = 0
        
        # Add a timer to periodically report status
        self.create_timer(5.0, self.report_status)

    def side_image_callback(self, msg: Image):
        """Display received side view images"""
        # Convert ROS image to OpenCV format
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.side_image = cv_image.copy()
        
        # Add frame info overlay
        self.side_frame_count += 1
        info_text = f"SIDE VIEW - Frame: {self.side_frame_count}, Size: {msg.width}x{msg.height}"
        
        # Add text overlay
        cv2.putText(cv_image, info_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Add timestamp
        timestamp = f"Time: {msg.header.stamp.sec}.{msg.header.stamp.nanosec//1000000:03d}"
        cv2.putText(cv_image, timestamp, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Show side view image
        cv2.imshow("Side View Camera", cv_image)
        
        # Update combined view
        self.update_combined_view()
        
        # Handle key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.get_logger().info("Quit requested")
            rclpy.shutdown()
            
        # Log reception
        if self.side_frame_count % 30 == 0:  # Every 30 frames (3 seconds at 10Hz)
            mean_brightness = np.mean(cv_image)
            self.get_logger().info(f"Side view: {self.side_frame_count} frames, brightness: {mean_brightness:.1f}")

    def top_image_callback(self, msg: Image):
        """Display received top view images"""
        try:
            self.get_logger().info(f" Received top view image: {msg.width}x{msg.height}, encoding: {msg.encoding}")
            
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.top_image = cv_image.copy()
            
            # Add frame info overlay
            self.top_frame_count += 1
            info_text = f"TOP VIEW - Frame: {self.top_frame_count}, Size: {msg.width}x{msg.height}"
            
            # Add text overlay
            cv2.putText(cv_image, info_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Add timestamp
            timestamp = f"Time: {msg.header.stamp.sec}.{msg.header.stamp.nanosec//1000000:03d}"
            cv2.putText(cv_image, timestamp, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Check if image is all black
            mean_brightness = np.mean(cv_image)
            if mean_brightness == 0:
                self.get_logger().warn(f"⚠️ Top view image is completely black (brightness: {mean_brightness})")
                # Add a visible indicator for black images
                cv2.putText(cv_image, "BLACK IMAGE DETECTED", (10, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Show top view image
            self.get_logger().info(" Displaying top view image in window")
            cv2.imshow("Top View Camera", cv_image)
            cv2.waitKey(1)  # Force window update
            
            # Update combined view
            self.update_combined_view()
            
            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.get_logger().info("Quit requested")
                rclpy.shutdown()
                
            # Log reception
            if self.top_frame_count % 10 == 0:  # Every 10 frames for more frequent updates
                self.get_logger().info(f"📷 Top view: {self.top_frame_count} frames, brightness: {mean_brightness:.1f}")
                
        except Exception as e:
            self.get_logger().error(f"Error processing top view image: {e}")
            import traceback
            self.get_logger().error(f"Traceback: {traceback.format_exc()}")
                
    def update_combined_view(self):
        """Create and display combined side-by-side view"""
        try:
            if self.side_image is not None and self.top_image is not None:
                # Resize images to same height for side-by-side display
                height = min(self.side_image.shape[0], self.top_image.shape[0])
                
                # Resize both images to same height while maintaining aspect ratio
                side_resized = cv2.resize(self.side_image, 
                                        (int(self.side_image.shape[1] * height / self.side_image.shape[0]), height))
                top_resized = cv2.resize(self.top_image, 
                                       (int(self.top_image.shape[1] * height / self.top_image.shape[0]), height))
                
                # Combine images horizontally
                combined = np.hstack((side_resized, top_resized))
                
                # Add labels
                cv2.putText(combined, "SIDE VIEW", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(combined, "TOP VIEW", (side_resized.shape[1] + 10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # Show combined view
                cv2.imshow("Combined View", combined)
                
        except Exception as e:
            self.get_logger().error(f"Error creating combined view: {e}")

    def report_status(self):
        """Report the status of both camera feeds"""
        self.get_logger().info(f"📊 Status - Side view: {self.side_frame_count} frames, Top view: {self.top_frame_count} frames")
        if self.side_frame_count == 0:
            self.get_logger().warn("⚠️ No side view frames received - check /camera/image topic")
        if self.top_frame_count == 0:
            self.get_logger().warn("⚠️ No top view frames received - check /camera/top_view topic")

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
