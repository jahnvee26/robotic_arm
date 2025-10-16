#!/usr/bin/env python3
"""
Pose Neutralizer Script

Reads octo_result.json and creates a new JSON file with neutralized positions.
This removes the effect of roll, pitch, yaw rotations from the position coordinates
by applying the inverse transformation matrix.
"""

import json
import numpy as np
import os

def rpy_to_rotmat(roll, pitch, yaw):
    """
    Convert roll, pitch, yaw to rotation matrix
    """
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])
    
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])
    
    return Rz @ Ry @ Rx

def neutralize_pose(pose_data):
    """
    Apply inverse transformation to neutralize the pose orientation effects
    """
    # Extract position and orientation
    pos = np.array([pose_data['x'], pose_data['y'], pose_data['z']])
    roll = pose_data['roll']
    pitch = pose_data['pitch'] 
    yaw = pose_data['yaw']
    
    # Get rotation matrix
    R = rpy_to_rotmat(roll, pitch, yaw)
    
    # Apply inverse transformation to get neutral position
    # The idea is: if the gripper was rotated, where would its center be
    # if it was in neutral orientation?
    R_inv = R.T  # Inverse of rotation matrix is its transpose
    
    # For gripper offset compensation, we might need to account for 
    # the gripper length/offset from wrist center
    # For now, we'll just apply the inverse rotation to the position
    neutral_pos = R_inv @ pos
    
    return {
        'x': float(neutral_pos[0]),
        'y': float(neutral_pos[1]), 
        'z': float(neutral_pos[2]),
        'original_x': float(pos[0]),
        'original_y': float(pos[1]),
        'original_z': float(pos[2]),
        'original_roll': roll,
        'original_pitch': pitch,
        'original_yaw': yaw,
        'neutralization_applied': True
    }

def process_octo_result(input_file, output_file):
    """
    Process octo_result.json and create neutralized version
    """
    # Read original file
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return False
    
    with open(input_file, 'r') as f:
        octo_data = json.load(f)
    
    print(f"📖 Processing {len(octo_data['target_poses'])} poses...")
    
    # Create neutralized version
    neutralized_data = {
        'original_file': input_file,
        'neutralization_method': 'inverse_rotation_matrix',
        'raw_action_shape': octo_data['raw_action_shape'],
        'timestep_count': octo_data['timestep_count'],
        'neutralized_poses': [],
        'original_poses': octo_data['target_poses']
    }
    
    # Process each pose
    for i, pose in enumerate(octo_data['target_poses']):
        print(f"  Processing timestep {i+1}/{len(octo_data['target_poses'])}...")
        
        # Show original pose
        print(f"    Original: x={pose['x']:.4f}, y={pose['y']:.4f}, z={pose['z']:.4f}, "
              f"roll={pose['roll']:.4f}, pitch={pose['pitch']:.4f}, yaw={pose['yaw']:.4f}")
        
        # Neutralize pose
        neutralized_pose = neutralize_pose(pose)
        neutralized_data['neutralized_poses'].append(neutralized_pose)
        
        # Show neutralized result
        print(f"    Neutral:  x={neutralized_pose['x']:.4f}, y={neutralized_pose['y']:.4f}, z={neutralized_pose['z']:.4f}")
        
        # Calculate difference
        dx = neutralized_pose['x'] - pose['x']
        dy = neutralized_pose['y'] - pose['y'] 
        dz = neutralized_pose['z'] - pose['z']
        print(f"    Delta:    Δx={dx:.4f}, Δy={dy:.4f}, Δz={dz:.4f}")
        print()
    
    # Save neutralized data
    with open(output_file, 'w') as f:
        json.dump(neutralized_data, f, indent=2)
    
    print(f"✅ Neutralized poses saved to: {output_file}")
    print(f"📊 Summary:")
    print(f"   - Original poses: {len(octo_data['target_poses'])}")
    print(f"   - Neutralized poses: {len(neutralized_data['neutralized_poses'])}")
    
    return True

def main():
    """
    Main function - process octo_result.json
    """
    input_file = '/tmp/octo_result.json'
    output_file = '/tmp/octo_result_neutralized.json'
    
    print("🔄 Pose Neutralizer - Removing orientation effects from positions")
    print("=" * 60)
    
    success = process_octo_result(input_file, output_file)
    
    if success:
        print("\n🎯 Neutralization complete!")
        print(f"   Input:  {input_file}")
        print(f"   Output: {output_file}")
        print("\nYou can now use the neutralized positions for robot control.")
    else:
        print("\n❌ Neutralization failed!")

if __name__ == '__main__':
    main()
