# MR-STORM Examples & Quickstart Scripts

This directory contains clean, standalone examples showing how to run **MR-STORM** across different simulators, real robot hardware, and distributed ROS 2 setups:

| Script | Description |
| :--- | :--- |
| [`01_gazebo_multi_arm.py`](./01_gazebo_multi_arm.py) | Connects MR-STORM to **Gazebo** (Harmonic / Ignition / Classic) via `ros2_control` (`JointTrajectoryController` or `ForwardPositionController`). Run with `--print_ros2_control_yaml` to auto-generate a Gazebo controller config. |
| [`02_real_robot_multi_arm.py`](./02_real_robot_multi_arm.py) | Runs MR-STORM on **physical robot arms** with real-time `SafetySupervisor` protection (joint/velocity/step clamping, EMA smoothing, E-Stop). Includes native Universal Robots (`ur_rtde`) and a 5-line callback wrapper for any robot SDK (Unitree G1, Franka, Kinova, etc.). |
| [`03_distributed_ros2_goal_sender.py`](./03_distributed_ros2_goal_sender.py) | Publishes crossing 6-DoF goal poses to **truly distributed ROS 2 MR-STORM arm nodes** and monitors the `/mr_storm/plan_broadcast` bus in real time. |
| [`04_custom_simulator_template.py`](./04_custom_simulator_template.py) | Minimal 25-line template showing how to plug MR-STORM into **MuJoCo, PyBullet, CoppeliaSim, or Drake** by subclassing `RobotBackend`. |

For full IROS 2026 paper benchmark suites in **NVIDIA Isaac Sim**, see [`projects_root/experiments/core_api/dataset_collector.py`](../projects_root/experiments/core_api/dataset_collector.py) and the main [`README.md`](../readme.md).
