#!/usr/bin/env python3
"""
Linear trajectory interpolation for joint motion
"""

import numpy as np

class LinearJointTrajectory:
    """Simple linear interpolation trajectory between joint positions"""
    
    def __init__(self, start_positions, end_positions, duration, joint_names=None):
        """
        Initialize linear trajectory
        
        Args:
            start_positions: Starting joint positions (numpy array or list)
            end_positions: Target joint positions (numpy array or list)
            duration: Duration of trajectory in seconds
            joint_names: List of joint names (optional)
        """
        self.start_positions = np.array(start_positions)
        self.end_positions = np.array(end_positions)
        self.duration = duration
        self.joint_names = joint_names or [f"joint_{i}" for i in range(len(start_positions))]
        
        # Calculate trajectory delta
        self.position_delta = self.end_positions - self.start_positions
        
    def get_position_at_time(self, t):
        """
        Get joint positions at time t
        
        Args:
            t: Time in seconds (0 to duration)
            
        Returns:
            Joint positions at time t
        """
        if t <= 0:
            return self.start_positions.copy()
        elif t >= self.duration:
            return self.end_positions.copy()
        else:
            # Linear interpolation
            progress = t / self.duration
            return self.start_positions + progress * self.position_delta
    
    def get_velocity_at_time(self, t):
        """
        Get joint velocities at time t
        
        Args:
            t: Time in seconds
            
        Returns:
            Joint velocities at time t
        """
        if t <= 0 or t >= self.duration:
            return np.zeros_like(self.start_positions)
        else:
            # Constant velocity for linear trajectory
            return self.position_delta / self.duration
    
    def get_trajectory_points(self, dt=0.01):
        """
        Get trajectory points sampled at dt intervals
        
        Args:
            dt: Time step in seconds
            
        Returns:
            List of (time, position, velocity) tuples
        """
        points = []
        t = 0.0
        while t <= self.duration:
            pos = self.get_position_at_time(t)
            vel = self.get_velocity_at_time(t)
            points.append((t, pos, vel))
            t += dt
        
        # Ensure we include the final point
        if points[-1][0] < self.duration:
            pos = self.get_position_at_time(self.duration)
            vel = self.get_velocity_at_time(self.duration)
            points.append((self.duration, pos, vel))
            
        return points
    
    def sample_trajectory(self, num_points):
        """
        Sample trajectory with a fixed number of points
        
        Args:
            num_points: Number of points to sample
            
        Returns:
            List of (time, position, velocity) tuples
        """
        if num_points < 2:
            raise ValueError("Need at least 2 points for trajectory")
            
        dt = self.duration / (num_points - 1)
        points = []
        
        for i in range(num_points):
            t = i * dt
            pos = self.get_position_at_time(t)
            vel = self.get_velocity_at_time(t)
            points.append((t, pos, vel))
            
        return points

class TrajectoryAction:
    """Action for a single trajectory point"""
    
    def __init__(self, joint_positions, joint_velocities=None, joint_efforts=None):
        self.joint_positions = np.array(joint_positions)
        self.joint_velocities = joint_velocities if joint_velocities is not None else np.zeros_like(self.joint_positions)
        self.joint_efforts = joint_efforts

def create_linear_trajectory_actions(start_positions, end_positions, duration, dt=1/60.0):
    """
    Create a sequence of trajectory actions for linear interpolation
    
    Args:
        start_positions: Starting joint positions
        end_positions: Target joint positions  
        duration: Duration in seconds
        dt: Time step (default 60 Hz)
        
    Returns:
        List of TrajectoryAction objects
    """
    traj = LinearJointTrajectory(start_positions, end_positions, duration)
    trajectory_points = traj.get_trajectory_points(dt)
    
    actions = []
    for t, pos, vel in trajectory_points:
        action = TrajectoryAction(pos, vel)
        actions.append(action)
        
    return actions
