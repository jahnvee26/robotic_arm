
import math
import numpy as np

class Kinematic:
    def __init__(self, phi_resolution: float = 1.0):
        # Robot arm dimensions in meters (converted from your original values)
        self.a0 = 0.0595   # Base height (5.95cm)
        self.a1 = 0.06959  # Base to shoulder joint (6.959cm)
        self.a1_horizontal = 0.04105  # horizontal component (4.105cm)
        self.a1_vertical = 0.05624    # vertical component (5.624cm)
        self.a2 = 0.169    # Shoulder to elbow length (16.9cm)
        self.a3 = 0.128    # Elbow to wrist length (12.8cm)
        self.a4 = 0.108    # Wrist to end effector length (10.8cm)

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


    def inverse_kinematics(self, target_position):
        px, py, pz = target_position
        
        solutions = []
        
        # Calculate base rotation angle
        theta0 = math.atan2(py, px)
        
        # Project to 2D 
        # Account for base height a0 and offset a1 from base to shoulder
        # Adjust the target position relative to the shoulder joint
        desired_wrist_x = np.sqrt(px ** 2 + py ** 2)

        if desired_wrist_x < self.a1_horizontal:
            print("Wrist cant reach the desired position, wrist will collide with arm")
            return None
        if pz < 0.0:
            print("Wrist cant reach the desired position, end effector with hit ground in downwards position")
            return None

        desired_wrist_z = pz + self.a4
        print(f"Desired wrist position: ({desired_wrist_x:.3f}, {desired_wrist_z:.3f}) m")
        # Iterate through different phi values (end-effector orientation)
        
        # Calculate elbow angle (theta2)
        # Subtract a1 offsets to get wrist position relative to shoulder
        wrist_x_dash = desired_wrist_x - self.a1_horizontal
        wrist_z_dash = desired_wrist_z - self.a0 - self.a1_vertical
        cos_theta2 = (wrist_x_dash** 2 + wrist_z_dash** 2 - self.a2 ** 2 - self.a3 ** 2) / (2 * self.a2 * self.a3)

        # Check if solution exists
        if abs(cos_theta2) >= 1:
            print("Wrist cant reach the desired position")
            return None
        else:
        
            theta2_1 = np.arccos(cos_theta2)
            theta2_2 = -np.arccos(cos_theta2)
            
            for theta2 in [theta2_1, theta2_2]:
                #TODO: check corner cases in beta calculation and alpha calculation
                beta = math.atan2(wrist_z_dash, wrist_x_dash)
                alpha = math.atan2(self.a3 * math.sin(theta2), self.a2 + self.a3 * math.cos(theta2))
                theta1 =  np.pi/2 - beta - alpha

                theta3 = np.pi - theta1 - theta2
                solutions.append([theta0, theta1, theta2, theta3, 0.0])
        # Only return the first solution for simplicity TODO: change if required in future
        return solutions[0]


    # def validate_joint_angles(self, joint_angles):

    #     for i, angle in enumerate(joint_angles[:6]):  # Check first 6 joints
    #         if i in self.joint_limits:
    #             min_angle, max_angle = self.joint_limits[i]
    #             if angle < min_angle or angle > max_angle:
    #                 return False
    #     return True

    def find_best_solution(self, solutions, 
                          current_angles):
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
    
        
    # def get_workspace_limits(self) -> dict:
    #     max_reach = self.a2 + self.a3 + self.a4
    #     min_reach = abs(self.a2 - self.a3) if self.a2 > self.a3 else 0
        
    #     return {
    #         'max_reach': max_reach,
    #         'min_reach': min_reach,
    #         'max_height': max_reach,
    #         'min_height': -max_reach
    #     }

    