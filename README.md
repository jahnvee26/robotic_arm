# robotic_arm
- 1) To directly publish joint angles
```
ros2 topic pub /target_joint_angles sensor_msgs/msg/JointState '{position: [0.0, 0.5, -0.5, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}' --once
```
- 2) To publish target position
```
ros2 topic pub /target_position geometry_msgs/msg/Point "{ x: 0.2, y: 0.1, z: 0.01}" --once
```

- 3) source the terminal where the issac-sim is running with ros2 inside the ~/jahnvee/IsaacSim-ros_Workspace
```
source build_ws/jazzy/jazzy_ws/install/local_setup.bash
source build_ws/jazzy/isaac_sim_ros_ws/install/local_setup.bash
```