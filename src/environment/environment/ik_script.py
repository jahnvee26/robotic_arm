import numpy as np

L0 = 0.0595   # base height (vertical) - base height is 59.5mm
L1 = 0.06959  # actual link length from base joint to shoulder joint: sqrt(41.05^2 + 56.24^2) = 69.59mm
L1_horizontal = 0.04105  # horizontal component (41.05mm)
L1_vertical = 0.05624   # vertical component (56.24mm)
L2 = 0.169    # shoulder to elbow (longer link)
L3 = 0.128    # elbow to wrist (shorter link)
L4 = 0.067    # wrist pitch link
L5 = 0.069    # wrist roll to EE


def isaac_sim_to_ik_coords(isaac_x, isaac_y, isaac_z):
    """
    Transform coordinates from Isaac Sim frame to IK solver frame.
    
    Isaac Sim → IK Script:
    - Isaac X → IK Y 
    - Isaac Y → IK -X
    - Isaac Z → IK Z (unchanged)
    
    Args:
        isaac_x, isaac_y, isaac_z: Coordinates in Isaac Sim frame
        
    Returns:
        tuple: (ik_x, ik_y, ik_z) in IK solver frame
    """
    ik_x = -isaac_y  # Isaac Y becomes IK -X
    ik_y = isaac_x   # Isaac X becomes IK Y
    ik_z = isaac_z   # Z unchanged
    return ik_x, ik_y, ik_z


def ik_to_isaac_sim_coords(ik_x, ik_y, ik_z):
    """
    Transform coordinates from IK solver frame to Isaac Sim frame.
    
    IK Script → Isaac Sim:
    - IK X → Isaac -Y
    - IK Y → Isaac X
    - IK Z → Isaac Z (unchanged)
    
    Args:
        ik_x, ik_y, ik_z: Coordinates in IK solver frame
        
    Returns:
        tuple: (isaac_x, isaac_y, isaac_z) in Isaac Sim frame
    """
    isaac_x = ik_y   # IK Y becomes Isaac X
    isaac_y = -ik_x  # IK X becomes Isaac -Y
    isaac_z = ik_z   # Z unchanged
    return isaac_x, isaac_y, isaac_z


# def dh_transform(theta, d, a, alpha):
#     """
#     Compute individual DH transformation matrix.
#     """
#     ct, st = np.cos(theta), np.sin(theta)
#     ca, sa = np.cos(alpha), np.sin(alpha)

#     return np.array([
#         [ct, -st*ca,  st*sa, a*ct],
#         [st,  ct*ca, -ct*sa, a*st],
#         [0,      sa,     ca,    d],
#         [0,       0,      0,    1]
#     ])

# def forward_kinematics_dh(joint_angles):
#     """
#     Compute forward kinematics using DH parameters.
#     Updated to match URDF with modified L1_to_L2 joint (rpy="-0.655391 0 0")
#     """
#     θ1, θ2, θ3, θ4, θ5 = joint_angles

#     # DH parameters adjusted to match Isaac Sim behavior
#     # Isaac Sim shows vertical arm when all joints = 0, so DH must predict this
#     dh_table = [
#         (θ1, L1, 0,   np.pi/2),                    # Joint 1: base rotation
#         (θ2 + np.pi/2, 0,  L2,  np.pi/2),         # Joint 2: +90° offset to match Isaac Sim
#         (θ3, 0,  L3,  0),                         # Joint 3: continue in Z direction
#         (θ4, 0,  L4,  0),                         # Joint 4: continue in Z direction
#         (θ5, 0,  L5,  0)                          # Joint 5: final reach in Z
#     ]

#     T = np.eye(4)
#     for params in dh_table:
#         T = T @ dh_transform(*params)

#     return T

class IK5DOF:
    def __init__(self):
        self.link_lengths = [L0, L1, L2, L3, L4, L5]
        self.L1_horizontal = L1_horizontal
        self.L1_vertical = L1_vertical

    def inverse_kinematics(self, target, pitch=0.0, roll=0.0):
        """
        Solve inverse kinematics for 5-DOF arm.
        
        Args:
            target (list): [x, y, z] in meters
            pitch (float): desired wrist pitch (rad)
            roll (float): desired wrist roll (rad)
        
        Returns:
            list: [θ1, θ2, θ3, θ4, θ5] in radians
        """
        x, y, z = target
        L0, L1, L2, L3, L4, L5 = self.link_lengths

        # 1) Base rotation (around vertical axis)
        theta1 = np.arctan2(y, x)

        # 2) Project to the plane containing the arm
        r_target = np.sqrt(x**2 + y**2)
        
        # Check if target is reachable considering the first link geometry
        if r_target < self.L1_horizontal:
            raise ValueError("Target too close - inside first link horizontal reach")

        # 3) Transform to shoulder joint coordinate system
        # The shoulder joint is offset by L1_horizontal horizontally and L1_vertical vertically
        # from the base joint position
        r_from_shoulder = r_target - self.L1_horizontal
        z_from_shoulder = z - L0 - self.L1_vertical

        # 4) Calculate wrist center position (accounting for desired pitch)
        wrist_len = L4 + L5
        
        # Wrist center position relative to shoulder joint
        r_wrist = r_from_shoulder - wrist_len * np.cos(pitch)
        z_wrist = z_from_shoulder - wrist_len * np.sin(pitch)

        # 5) Distance from shoulder to wrist center
        D = np.sqrt(r_wrist**2 + z_wrist**2)

        # Check reachability
        if D > (L2 + L3) or D < abs(L2 - L3):
            raise ValueError(f"Target not reachable. Distance {D:.4f}m, max reach {L2 + L3:.4f}m")

        # 6) Elbow angle using law of cosines
        cos_theta3 = (L2**2 + L3**2 - D**2) / (2 * L2 * L3)
        if abs(cos_theta3) > 1:
            raise ValueError("Target not reachable - elbow angle calculation failed")
        
        # Choose elbow-up configuration (positive angle)
        theta3 = np.arccos(cos_theta3)

        # 7) Shoulder angle
        # Angle from shoulder to wrist center
        gamma = np.arctan2(z_wrist, r_wrist)
        
        # Angle in triangle from shoulder to wrist center
        cos_alpha = (L2**2 + D**2 - L3**2) / (2 * L2 * D)
        if abs(cos_alpha) > 1:
            raise ValueError("Target not reachable - shoulder angle calculation failed")
        alpha = np.arccos(cos_alpha)
        
        # Shoulder joint angle (measured from horizontal)
        theta2 = gamma - alpha

        # 8) Wrist pitch angle
        # The wrist should maintain the desired pitch relative to horizontal
        # Current arm angle at wrist is (theta2 + theta3)
        theta4 = pitch - (theta2 + theta3)

        # 9) Wrist roll
        theta5 = roll

        return [theta1, theta2, theta3, theta4, theta5]

    def inverse_kinematics_isaac_sim(self, isaac_target, pitch=0.0, roll=0.0):
        """
        Solve inverse kinematics for Isaac Sim coordinates.
        
        This is a wrapper that handles coordinate system transformation:
        - Takes target in Isaac Sim coordinate frame
        - Converts to IK solver frame
        - Solves IK
        - Returns joint angles for Isaac Sim
        
        Args:
            isaac_target (list): [x, y, z] in Isaac Sim coordinate frame
            pitch (float): desired wrist pitch (rad)
            roll (float): desired wrist roll (rad)
        
        Returns:
            list: [θ1, θ2, θ3, θ4, θ5] in radians for Isaac Sim
        """
        isaac_x, isaac_y, isaac_z = isaac_target
        
        # Transform from Isaac Sim coordinates to IK solver coordinates
        ik_x, ik_y, ik_z = isaac_sim_to_ik_coords(isaac_x, isaac_y, isaac_z)
        
        # Solve IK in the IK solver's coordinate frame
        joint_angles = self.inverse_kinematics([ik_x, ik_y, ik_z], pitch, roll)
        
        return joint_angles
