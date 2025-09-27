"""
The robot arm has 5 degrees of freedom:
- Joint 0: Base rotation (around Z-axis
- Joint 1: Shoulder pitch
- Joint 2: Elbow pitch  
- Joint 3: Wrist pitch
- Joint 4: Wrist roll
"""

import math
from py_compile import main
from typing import List, Tuple, Optional, Union
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class Kinematic:
    """
    Handles forward and inverse kinematics for the ROT3U robotic arm.
    
    The robot arm uses Denavit-Hartenberg (DH) parameters for kinematic calculations.
    Link lengths are specified in centimeters.
    """
    
    def __init__(self, phi_resolution: float = 1.0):
        """
        Initialize the Kinematic class.
        
        Args:
            phi_resolution (float): Resolution for phi angle calculations in degrees
        """
        # Robot arm dimensions in cm
        self.a0 = 0.0595 # Base height
        self.a1 = 0.06959  # actual  from base joint to shoulder joint: sqrt(41.05^2 + 56.24^2) = 69.59mm
        self.a1_horizontal = 0.04105  # horizontal component (41.05mm)
        self.a1_vertical = 0.05624   # vertical component (56.24mm)
        self.a2 = 10.5  # Shoulder to elbow length
        self.a3 = 10.0  # Elbow to wrist length  
        self.a4 = 16.8  # Wrist to end effector length
        
        # Current target coordinates
        self.coord = [0.0, 0.0, 0.0]
        
        # Phi angle resolution for inverse kinematics
        self.phi_resolution = phi_resolution
        self.phi_deg_values = [0]  # Currently only using phi = 0
        
        # Denavit-Hartenberg parameters
        self.dh_params = [
            {'alpha': 0, 'a': 0, 'd': 0},                    # Base joint
            {'alpha': math.pi / 2, 'a': 0, 'd': 0},          # Shoulder joint
            {'alpha': math.pi, 'a': self.a2, 'd': 0},        # Elbow joint
            {'alpha': 0, 'a': self.a3, 'd': 0},              # Wrist joint
            {'alpha': 0, 'a': 0, 'd': 0},                    # End effector
        ]
        
        # Joint limits in degrees
        self.joint_limits = {
            0: (-180, 180),    # Base rotation
            1: (-20, 100),    # Shoulder pitch
            2: (-154.699, 154.699),    # Elbow pitch
            3: (-120.321, 120.321),    # Wrist pitch
            4: (-89.95, 0),    # Wrist roll
        }


    def inverse_kinematics(self, target_position: List[float]) -> List[List[float]]:
        """
        Calculate inverse kinematics to get joint angles from end-effector position.
        
        Args:
            target_position (List[float]): Target [x, y, z] coordinates in cm
            
        Returns:
            List[List[float]]: List of possible joint angle solutions in radians
        """
        px, py, pz = target_position
        self.coord = target_position
        
        solutions = []
        
        # Calculate base rotation angle
        theta0 = math.atan2(py, px)
        
        # Project to 2D (remove base rotation effect)
        px_projected = px * math.cos(theta0) + py * math.sin(theta0)
        px = px_projected
        # Account for base height a0 and offset a1 from base to shoulder
        # Adjust the target position relative to the shoulder joint
        px_eff = px - self.a1_horizontal
        pz_eff = pz - self.a0 - self.a1_vertical

        # Iterate through different phi values (end-effector orientation)
        for phi_deg in self.phi_deg_values:
            phi = math.radians(phi_deg)
            
            # Calculate wrist center position
            wx = px_eff - (self.a4 * np.cos(phi))
            wz = pz_eff - (self.a4 * np.sin(phi))
            
            # Calculate elbow angle (theta2)
            cos_theta2 = (wx ** 2 + wz ** 2 - self.a2 ** 2 - self.a3 ** 2) / (2 * self.a2 * self.a3)
            
            # Check if solution exists
            if abs(cos_theta2) <= 1:
                # Two possible elbow configurations
                sin_theta2_1 = np.sqrt(1 - cos_theta2 ** 2)
                sin_theta2_2 = -np.sqrt(1 - cos_theta2 ** 2)
                
                theta2_1 = np.arctan2(sin_theta2_1, cos_theta2)
                theta2_2 = np.arctan2(sin_theta2_2, cos_theta2)
                
                # Calculate shoulder angles for both elbow configurations
                for theta2 in [theta2_1, theta2_2]:
                    denom = self.a2 ** 2 + self.a3 ** 2 + 2 * self.a2 * self.a3 * np.cos(theta2)
                    
                    if denom > 0:
                        sin_theta1 = (wz * (self.a2 + self.a3 * np.cos(theta2)) - 
                                     self.a3 * np.sin(theta2) * wx) / denom
                        cos_theta1 = (wx * (self.a2 + self.a3 * np.cos(theta2)) + 
                                     self.a3 * np.sin(theta2) * wz) / denom
                        
                        theta1 = np.arctan2(sin_theta1, cos_theta1)
                        
                        # Calculate wrist angle
                        theta3 = phi - theta1 - theta2
                        
                        # Create solution [base, shoulder, elbow, wrist]
                        solution = [theta0, theta1, -theta2, -theta3 + (math.pi / 2)]
                        solutions.append(solution)
        
        return solutions

    def get_end_effector_position(self, joint_angles: List[float]) -> Tuple[float, float, float]:
        """
        Get the end-effector position from joint angles.
        
        Args:
            joint_angles (List[float]): Joint angles in radians
            
        Returns:
            Tuple[float, float, float]: End-effector position (x, y, z) in cm
        """
        transform = self.forward_kinematics(joint_angles)
        return transform[0, 3], transform[1, 3], transform[2, 3]

    def validate_joint_angles(self, joint_angles: List[float]) -> bool:
        """
        Validate that all joint angles are within their limits.
        
        Args:
            joint_angles (List[float]): Joint angles in degrees
            
        Returns:
            bool: True if all angles are valid, False otherwise
        """
        for i, angle in enumerate(joint_angles[:6]):  # Check first 6 joints
            if i in self.joint_limits:
                min_angle, max_angle = self.joint_limits[i]
                if angle < min_angle or angle > max_angle:
                    return False
        return True

    def find_best_solution(self, solutions: List[List[float]], 
                          current_angles: List[float]) -> Optional[List[float]]:
        """
        Find the best inverse kinematics solution based on proximity to current angles.
        
        Args:
            solutions (List[List[float]]): List of possible solutions
            current_angles (List[float]): Current joint angles in radians
            
        Returns:
            Optional[List[float]]: Best solution in degrees, or None if no valid solution
        """
        valid_solutions = []
        
        for solution in solutions:
            # Convert to degrees
            solution_degrees = [math.degrees(angle) for angle in solution]
            
            # Add gripper angles (set to 0 for now)
            full_solution = [0] + solution_degrees + [0]
            
            # Check if solution is within joint limits
            if self.validate_joint_angles(full_solution):
                valid_solutions.append(solution_degrees)
        
        if not valid_solutions:
            return None
        
        # Find solution closest to current angles
        if len(current_angles) >= 4:
            current_subset = current_angles[1:5]  # Compare only the main 4 joints
            min_distance = float('inf')
            best_solution = None
            
            for solution in valid_solutions:
                distance = sum(abs(a - b) for a, b in zip(solution, current_subset))
                if distance < min_distance:
                    min_distance = distance
                    best_solution = solution
            
            return best_solution
        else:
            return valid_solutions[0]
    
    def update_phi_resolution(self, resolution: float) -> None:
        """
        Update the phi angle resolution for inverse kinematics.
        
        Args:
            resolution (float): New resolution in degrees
        """
        self.phi_resolution = resolution
        self.phi_deg_values = [0]  # Currently only using phi = 0
        
    def get_workspace_limits(self) -> dict:
        """
        Calculate the workspace limits of the robot arm.
        
        Returns:
            dict: Dictionary containing workspace limits
        """
        max_reach = self.a2 + self.a3 + self.a4
        min_reach = abs(self.a2 - self.a3) if self.a2 > self.a3 else 0
        
        return {
            'max_reach': max_reach,
            'min_reach': min_reach,
            'max_height': max_reach,
            'min_height': -max_reach
        }

    def is_position_reachable(self, position: List[float]) -> bool:
        """
        Check if a position is within the robot's workspace.
        
        Args:
            position (List[float]): Target position [x, y, z]
            
        Returns:
            bool: True if position is reachable, False otherwise
        """
        x, y, z = position
        distance = math.sqrt(x**2 + y**2 + z**2)
        workspace = self.get_workspace_limits()
        
        return workspace['min_reach'] <= distance <= workspace['max_reach']
    
def main():
# Simple test with hardcoded values
ik_solver = Kinematic()

# Test target position (x, y, z in centimeters - matching the Kinematic class units)
target = [20.0, 10.0, 15.0]  # 20cm, 10cm, 15cm

print(f"Target position: {target} cm")


joint_angles_solutions = ik_solver.inverse_kinematics(target)
print(f"Found {len(joint_angles_solutions)} solution(s)")

if joint_angles_solutions:
    for i, solution in enumerate(joint_angles_solutions):
        print(f"Solution {i+1}:")
        print(f"  Joint angles (radians): {solution}")
        print(f"  Joint angles (degrees): {[np.degrees(angle) for angle in solution]}")
else:
    print("No solutions found!")
    
if __name__ == "__main__":
    main()
