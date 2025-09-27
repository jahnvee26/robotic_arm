#!/usr/bin/env python3
"""
Keyboard Control Node for Robot Arm

This node allows manual control of the robot arm target position using arrow keys.
Uses pygame for keyboard input handling.

Controls:
- Arrow Up: Move forward (+Y)
- Arrow Down: Move backward (-Y)
- Arrow Left: Move left (-X)
- Arrow Right: Move right (+X)
- Page Up: Move up (+Z)
- Page Down: Move down (-Z)
- Space: Reset to center position
- ESC/q: Quit
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
import pygame
import sys
import threading
import time

class KeyboardControlNode(Node):
    """
    ROS2 Node for keyboard-based robot arm control using pygame
    
    Publishes to: /target_position (geometry_msgs/Point)
    """

    def __init__(self):
        super().__init__('keyboard_control_node')
        
        # Initialize pygame with error handling
        pygame.init()
        
        
        try:
            # Create a small window for pygame (required for event handling)
            self.screen = pygame.display.set_mode((400, 300), pygame.DOUBLEBUF)
            pygame.display.set_caption("Robot Arm Keyboard Control")
            self.display_available = True
        except Exception as e:
            self.get_logger().warn(f"Could not create pygame display: {e}")
            self.get_logger().info("Running in headless mode - keyboard events only")
            self.screen = None
            self.display_available = False
        
        # Create publisher for target position
        self.position_publisher = self.create_publisher(
            Point,
            '/target_position',
            10
        )
        
        # Current target position (start at a safe position)
        self.current_position = Point()
        self.current_position.x = 0.2  # 20cm forward
        self.current_position.y = 0.0  # centered
        self.current_position.z = 0.1  # 10cm above base
        
        # Movement step size (in meters)
        self.step_size = 0.005  # 0.5cm steps

        # Workspace limits (in meters)
        self.limits = {
            'x': (0.05, 0.4),   # 5cm to 40cm
            'y': (-0.3, 0.3),   # -30cm to +30cm  
            'z': (0.02, 0.3)    # 2cm to 30cm
        }
        
        # Control flags
        self.running = True
        self.publish_rate = 10.0  # Hz
        
        # Key states for continuous movement
        self.keys_pressed = {
            pygame.K_UP: False,
            pygame.K_DOWN: False,
            pygame.K_LEFT: False,
            pygame.K_RIGHT: False,
            pygame.K_PAGEUP: False,
            pygame.K_PAGEDOWN: False
        }
        
        # Start the pygame event loop in a separate thread
        self.pygame_thread = threading.Thread(target=self.pygame_event_loop)
        self.pygame_thread.daemon = True
        self.pygame_thread.start()
        
        # Start the movement processing timer
        self.movement_timer = self.create_timer(1.0 / 10.0, self.process_movement)  # 10Hz for smooth movement

        # Start the publishing timer
        self.publish_timer = self.create_timer(1.0 / self.publish_rate, self.publish_position)
        
        self.get_logger().info("Keyboard Control Node started!")
        self.print_instructions()
        
        # Publish initial position
        self.publish_position()

    def print_instructions(self):
        """Print control instructions."""
        instructions = """
╔══════════════════════════════════════════════════╗
║                KEYBOARD CONTROLS                 ║
╠══════════════════════════════════════════════════╣
║  Arrow Up    : Move Forward  (+Y)                ║
║  Arrow Down  : Move Backward (-Y)                ║
║  Arrow Left  : Move Left     (-X)                ║
║  Arrow Right : Move Right    (+X)                ║
║  Page Up     : Move Up       (+Z)                ║
║  Page Down   : Move Down     (-Z)                ║
║  Space       : Reset to center position          ║
║  ESC / q     : Quit                              ║
╠══════════════════════════════════════════════════╣
║  Step size: {:.3f}m ({:.1f}cm)                      ║
║  Current position: X={:.3f}, Y={:.3f}, Z={:.3f}     ║
║                                                  ║
║  Focus the pygame window to use controls!        ║
╚══════════════════════════════════════════════════╝
        """.format(self.step_size, self.step_size * 100,
                   self.current_position.x, self.current_position.y, self.current_position.z)
        print(instructions)

    def draw_interface(self):
        """Draw the pygame interface."""
        if not self.display_available or self.screen is None:
            return
            
        try:
            # Fill background with dark color
            self.screen.fill((20, 20, 20))
            # No need for display.flip() since we're not showing anything
        except pygame.error as e:
            self.get_logger().warn(f"Display update failed: {e}")
            self.display_available = False

    def pygame_event_loop(self):
        """Handle pygame events in a separate thread."""
        clock = pygame.time.Clock()
        self.get_logger().info('Pygame event loop started')
        
        while self.running:
            try:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.quit()
                        
                    elif event.type == pygame.KEYDOWN:
                        if event.key in self.keys_pressed:
                            self.keys_pressed[event.key] = True
                        elif event.key == pygame.K_SPACE:
                            self.reset_position()
                        elif event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                            self.quit()
                            
                    elif event.type == pygame.KEYUP:
                        if event.key in self.keys_pressed:
                            self.keys_pressed[event.key] = False
                
                # No need to update display since we're only capturing input
                clock.tick(30)  # 30 FPS for pygame
                
            except Exception as e:
                self.get_logger().error(f"Pygame event loop error: {e}")
                break

    def process_movement(self):
        """Process continuous movement based on currently pressed keys."""
        moved = False
        
        if self.keys_pressed[pygame.K_UP]:
            if self.move_position('y', self.step_size):
                moved = True
        if self.keys_pressed[pygame.K_DOWN]:
            if self.move_position('y', -self.step_size):
                moved = True
        if self.keys_pressed[pygame.K_RIGHT]:
            if self.move_position('x', self.step_size):
                moved = True
        if self.keys_pressed[pygame.K_LEFT]:
            if self.move_position('x', -self.step_size):
                moved = True
        if self.keys_pressed[pygame.K_PAGEUP]:
            if self.move_position('z', self.step_size):
                moved = True
        if self.keys_pressed[pygame.K_PAGEDOWN]:
            if self.move_position('z', -self.step_size):
                moved = True

    def move_position(self, axis, delta):
        """Move the target position along the specified axis."""
        moved = False
        
        if axis == 'x':
            new_value = self.current_position.x + delta
            if self.limits['x'][0] <= new_value <= self.limits['x'][1]:
                self.current_position.x = new_value
                moved = True
            else:
                if delta > 0:
                    self.get_logger().warn(f"X position at maximum limit: {self.limits['x'][1]}")
                else:
                    self.get_logger().warn(f"X position at minimum limit: {self.limits['x'][0]}")
                
        elif axis == 'y':
            new_value = self.current_position.y + delta
            if self.limits['y'][0] <= new_value <= self.limits['y'][1]:
                self.current_position.y = new_value
                moved = True
            else:
                if delta > 0:
                    self.get_logger().warn(f"Y position at maximum limit: {self.limits['y'][1]}")
                else:
                    self.get_logger().warn(f"Y position at minimum limit: {self.limits['y'][0]}")
                
        elif axis == 'z':
            new_value = self.current_position.z + delta
            if self.limits['z'][0] <= new_value <= self.limits['z'][1]:
                self.current_position.z = new_value
                moved = True
            else:
                if delta > 0:
                    self.get_logger().warn(f"Z position at maximum limit: {self.limits['z'][1]}")
                else:
                    self.get_logger().warn(f"Z position at minimum limit: {self.limits['z'][0]}")
        
        if moved:
            self.get_logger().info(f"Position: X={self.current_position.x:.3f}, Y={self.current_position.y:.3f}, Z={self.current_position.z:.3f}")
        
        return moved

    def reset_position(self):
        """Reset to the center/safe position."""
        self.current_position.x = 0.2
        self.current_position.y = 0.0
        self.current_position.z = 0.1
        self.get_logger().info("Position reset to center")

    def publish_position(self):
        """Publish the current target position."""
        self.position_publisher.publish(self.current_position)

    def quit(self):
        """Quit the node gracefully."""
        self.running = False
        self.get_logger().info("Shutting down keyboard control...")
        pygame.quit()
        rclpy.shutdown()

    def __del__(self):
        """Destructor to ensure pygame is cleaned up."""
        try:
            pygame.quit()
        except:
            pass


def main(args=None):
    rclpy.init(args=args)
    
    try:
        keyboard_node = KeyboardControlNode()
        rclpy.spin(keyboard_node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        try:
            pygame.quit()
        except:
            pass
        rclpy.shutdown()


if __name__ == '__main__':
    main()