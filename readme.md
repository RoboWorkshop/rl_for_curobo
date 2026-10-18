# MR-STORM: Scalable Multi-Arm Control By Distributed MPC

[![IEEE/RSJ IROS 2026](https://img.shields.io/badge/IEEE%2FRSJ-IROS%202026-00629B.svg)](https://roboworkshop.github.io/multi-robot-mpc/)
[![arXiv](https://img.shields.io/badge/arXiv-2602.10114-b31b1b.svg)](https://arxiv.org/abs/2602.10114)
[![Project Page](https://img.shields.io/badge/Project-Website-2ea44f.svg)](https://roboworkshop.github.io/multi-robot-mpc/)
[![Simulators](https://img.shields.io/badge/Simulators-Isaac%20Sim%20%7C%20Gazebo%20%7C%20MuJoCo-orange.svg)](#multi-simulator--real-hardware-quickstart)
[![ROS 2](https://img.shields.io/badge/ROS%202-Distributed%20Multi--Arm-22314E.svg)](#3-distributed-multi-arm-with-ros-2)

<p align="center">
  <img width="100%" alt="MR-STORM Banner" src="https://github.com/user-attachments/assets/296e01e6-7869-45dd-bd94-4688dae235bc" />
</p>

> **Official Repository for the IEEE/RSJ IROS 2026 Paper:**  
> **[MR-STORM: Scalable Multi-Arm Control By Distributed MPC](https://arxiv.org/abs/2602.10114)**  
> [Dan Evron](mailto:evrond@post.bgu.ac.il), [Elias Goldsztejn](mailto:eliasgol@post.bgu.ac.il), [Dan R. Suissa](mailto:danrouve@post.bgu.ac.il), and [Ronen I. Brafman](mailto:brafman@bgu.ac.il)  
> *Foundations of AI Institute, Stein Faculty of Computer and Information Science, Ben-Gurion University of the Negev*  
> *(Note: `rl_for_curobo` is the legacy GitHub repository URL slug; the Python package provides the `mr_storm` API directly.)*

---

## Overview

Centralized motion planning algorithms struggle to scale in multi-arm manipulation, where tight shared workspaces and inter-arm coupling cause the joint state space to grow exponentially with the number of arms.

**MR-STORM** is a decentralized, GPU-accelerated Model Predictive Path Integral (MPPI) framework building on [STORM](https://arxiv.org/abs/2104.13542) and [CuRobo](https://curobo.org/) that controls multiple robotic arms in real time (30–100+ Hz):

1. **Decentralized & Scalable:** Each robotic arm runs its own local GPU MPPI trajectory optimizer, scaling linearly in per-agent computation to 4+ high-DoF manipulators in tight workspaces.
2. **Horizon Plan-Sharing:** At every control cycle, each arm integrates its MPPI mean acceleration policy over the horizon $H$ and broadcasts its predicted task-space collision spheres $\{(p_{t,s}, r_s)\}_{t=1..H}$ in the shared world frame.
3. **Dynamic Priority Conflict Resolution ($\tau$):** To prevent livelocks and deadlocks in symmetric bottlenecks, each arm weights inter-arm collision costs by relative goal proximity:
   $$d_{\text{eff}}(i, j) = d(i, j) \cdot \left(\frac{e_{\text{goal}}(j)}{e_{\text{goal}}(i)}\right)^{\tau}$$
   where $\tau \ge 0$ (`wta_trust`) smoothly prioritizes arms closer to completing their current task.
4. **Simulator- & Hardware-Agnostic (`mr_storm`):** Run with **NVIDIA Isaac Sim**, **Gazebo (Harmonic / Ignition / Classic)**, **MuJoCo / PyBullet**, **Distributed ROS 2 nodes**, or directly on **physical robots** (Unitree G1, Universal Robots UR5e/UR10e, Franka Emika).

---

## Repository Structure

```text
├── mr_storm/                       # Simulator- & hardware-agnostic MR-STORM package
│   ├── core/
│   │   ├── agent.py                # MRStormAgent: standalone single-arm GPU MPPI + horizon plan rollout
│   │   ├── coordinator.py          # MRStormCoordinator: multi-arm plan-board & priority coordinator
│   │   └── types.py                # RobotJointState, JointCommand, BroadcastPlanPacket, ArmConfig
│   ├── backends/
│   │   ├── base.py                 # Abstract RobotBackend interface (plug in any simulator or SDK)
│   │   ├── gazebo_backend.py       # GazeboROS2Backend (ros2_control trajectory/position/velocity)
│   │   ├── real_robot_backend.py   # RealRobotBackend + SafetySupervisor + UR RTDE / custom SDK drivers
│   │   └── kinematic_backend.py    # Lightweight simulator-free backend for rapid testing & CI
│   └── ros2/
│       ├── distributed_arm_node.py # True distributed ROS 2 node per arm + /mr_storm/plan_broadcast bus
│       ├── config/                 # Multi-arm fleet configs (quad_ur5e_fleet.yaml, dual_franka_fleet.yaml)
│       └── launch/                 # ROS 2 launch files for N distributed arm nodes
├── examples/                       # Ready-to-run quickstart scripts (Gazebo, Real Robot, ROS 2, Custom Sim)
├── curobo/                         # Bundled CuRobo fork with custom DynamicObsCost GPU rollout kernels
└── projects_root/                  # IROS 2026 Isaac Sim benchmark suite, configs, and analysis tools
    ├── experiments/core_api/       # dataset_collector.py & benchmark_sim.py
    └── projects/dynamic_obs/       # DynamicObsCollPredictor & runtime_topics
```

---

## Installation

### System Requirements
- **OS:** Ubuntu $\ge$ 20.04
- **GPU:** NVIDIA GPU with CUDA (8+ GB VRAM; RTX 4060 or higher recommended)
- **Tools:** `git`, `git-lfs`, `conda`

### Step 1: Create Conda Environment & Install PyTorch / CuRobo

```bash
# Python 3.10 (for Isaac Sim 4.5 / ROS 2 Humble) or Python 3.11 (for Isaac Sim 5.0 / ROS 2 Jazzy)
conda create -n mr_storm python=3.10 -y
conda activate mr_storm

# Clone the repository and pull LFS assets (robot meshes & USD scenes)
git clone https://github.com/RoboWorkshop/rl_for_curobo.git
cd rl_for_curobo
git lfs install
git lfs pull

# Install bundled CuRobo (includes MR-STORM's custom GPU DynamicObsCost kernel)
cd curobo
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO=0.0.0+local pip install -e . --no-build-isolation
cd ..

# Install the mr_storm package in editable mode
pip install -e .
```

### Step 2 (Optional): Install NVIDIA Isaac Sim (For Paper Benchmarks)

If you want to run the **Isaac Sim** benchmark suite (Gazebo, Real Robot, and Custom Simulator backends do **not** require Isaac Sim):

```bash
# Option A: Isaac Sim 4.5 (Python 3.10, glibc/ldd >= 2.34)
pip install isaacsim[all,extscache]==4.5.0 --extra-index-url https://pypi.nvidia.com

# Option B: Isaac Sim 5.0 (Python 3.11, glibc/ldd >= 2.35)
pip install isaacsim[all,extscache]==5.0.0 --extra-index-url https://pypi.nvidia.com
```

---

## Multi-Simulator & Real Hardware Quickstart

### 1. Gazebo Simulator (Harmonic / Ignition / Classic via `ros2_control`)

[`GazeboROS2Backend`](mr_storm/backends/gazebo_backend.py) connects MR-STORM directly to Gazebo's `ros2_control` controllers (`JointTrajectoryController` or `ForwardPositionController`).

```bash
# Generate a ready-to-use ros2_control YAML configuration for your Gazebo multi-arm world:
python3 examples/01_gazebo_multi_arm.py --print_ros2_control_yaml > my_gazebo_controllers.yaml

# Run MR-STORM against a running Gazebo multi-arm simulation:
python3 examples/01_gazebo_multi_arm.py \
  --fleet_config mr_storm/ros2/config/quad_ur5e_fleet.yaml \
  --command_interface trajectory
```

### 2. Real Robot Hardware (Any Physical Robot Arm)

[`RealRobotBackend`](mr_storm/backends/real_robot_backend.py) wraps physical manipulators with a real-time [`SafetySupervisor`](mr_storm/backends/real_robot_backend.py) that enforces per-cycle step clamping ($|\Delta q| \le \Delta q_{\max}$), joint position/velocity/acceleration limits, EMA command smoothing, and instant E-Stop.

- **Universal Robots (UR3e / UR5e / UR10e) via `ur_rtde`:**
  ```bash
  pip install ur-rtde
  python3 examples/02_real_robot_multi_arm.py --mode ur_rtde --arm0_ip 192.168.1.101 --arm1_ip 192.168.1.102
  ```
- **Any Custom Robot SDK (Unitree G1, Franka, Kinova, CAN bus) in 5 lines of Python:**
  ```python
  from mr_storm import ArmConfig, MRStormCoordinator, RealRobotBackend, CallbackHardwareDriver

  coordinator = MRStormCoordinator([
      ArmConfig(robot_id=0, robot_yml="ur5e.yml", base_pose=[0.0, -0.25, 0.0, 1.0, 0.0, 0.0, 0.0]),
      ArmConfig(robot_id=1, robot_yml="ur5e.yml", base_pose=[0.0,  0.25, 0.0, 1.0, 0.0, 0.0, 0.0]),
  ])
  drivers = {
      0: CallbackHardwareDriver(coordinator.agents[0].joint_names, read_fn=arm0.get_q_qd, write_fn=arm0.servo_q_qd),
      1: CallbackHardwareDriver(coordinator.agents[1].joint_names, read_fn=arm1.get_q_qd, write_fn=arm1.servo_q_qd),
  }
  backend = RealRobotBackend(drivers=drivers, control_freq_hz=30.0)
  ```

### 3. Distributed Multi-Arm with ROS 2

In distributed mode, **each robotic arm runs as its own independent OS process or on its own dedicated GPU / PC**. Arms exchange compact binary `BroadcastPlanPacket` messages over the `/mr_storm/plan_broadcast` DDS topic (with automatic latency horizon-shifting) and publish 3D horizon collision sphere markers to RViz2 (`/arm_<i>/horizon_spheres`).

```bash
# Launch 4 independent decentralized MR-STORM arm nodes:
ros2 launch mr_storm/ros2/launch/distributed_multi_arm.launch.py \
  num_arms:=4 \
  fleet_config:=mr_storm/ros2/config/quad_ur5e_fleet.yaml

# Or launch a single arm node on a dedicated machine/GPU (e.g. Arm 0 on PC 1, Arm 1 on PC 2):
python3 mr_storm/ros2/distributed_arm_node.py --robot_id 0 --fleet_config mr_storm/ros2/config/quad_ur5e_fleet.yaml

# Send crossing 6-DoF goal poses and monitor the /mr_storm/plan_broadcast bus:
python3 examples/03_distributed_ros2_goal_sender.py --num_arms 4
```

### 4. Plugging Into Other Simulators (MuJoCo, PyBullet, CoppeliaSim, Drake)

Subclass [`RobotBackend`](mr_storm/backends/base.py) and implement two methods (`get_joint_states` and `send_joint_commands`). See [`examples/04_custom_simulator_template.py`](examples/04_custom_simulator_template.py) for a complete 25-line working example.

---

## Reproducing the IROS 2026 Paper Benchmarks (Isaac Sim)

All simulation benchmarks from the paper are configured via [`projects_root/experiments/benchmarks/cfgs/combo_cfg.yml`](projects_root/experiments/benchmarks/cfgs/combo_cfg.yml) and executed through [`dataset_collector.py`](projects_root/experiments/core_api/dataset_collector.py).

### Supported Algorithms (`alg`)

| Code | Algorithm | Plan Broadcasting | Priority ($\tau$) | Description |
| :---: | :--- | :---: | :---: | :--- |
| **`O`** | **MR-STORM (Ours)** | Full Horizon ($H$) | $\tau > 0$ (`pose_wta`) | Decentralized GPU MPPI with horizon plan-sharing & dynamic priority |
| **`O-`** | **MR-STORM ($\tau=0$)** | Full Horizon ($H$) | $\tau = 0$ (Equal weight) | Ablation without priority conflict resolution |
| **`SD`** | **STORM Decentralized** | Static Current Pose | None | Decentralized MPPI treating peers as static obstacles at their current pose |
| **`SC`** | **STORM Centralized** | N/A (Joint Space) | N/A | Single centralized MPPI planning over the concatenated DoFs of all arms |
| **`CC`** | **CuRobo Centralized** | N/A (Joint Space) | N/A | Centralized CuMotion graph + trajectory optimization baseline |
| **`D`** | **dRRT\*** | N/A | N/A | Sampling-based multi-robot RRT* baseline |

### Running Benchmark Experiments

```bash
# 1. Inspect available CLI options
python3 projects_root/experiments/core_api/dataset_collector.py --help

# 2. Run with the default combo_cfg.yml (GUI mode)
python3 projects_root/experiments/core_api/dataset_collector.py

# 3. Run a custom combo config file in headless mode (e.g. on a server/cluster)
python3 projects_root/experiments/core_api/dataset_collector.py \
  --combo_cfg_path projects_root/experiments/benchmarks/cfgs/combo_cfg.yml \
  --vis_mode headless
```

---

## Citation

If you find **MR-STORM** useful in your research or applications, please cite our IEEE/RSJ IROS 2026 paper:

```bibtex
@inproceedings{evron2026mrstorm,
  title     = {MR-STORM: Scalable Multi-Arm Control By Distributed MPC},
  author    = {Evron, Dan and Goldsztejn, Elias and Suissa, Dan R. and Brafman, Ronen I.},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2602.10114}
}
```

---

## Additional Resources & Acknowledgements
- **[MR-STORM Project Website & Videos](https://roboworkshop.github.io/multi-robot-mpc/)**
- **[arXiv Paper (2602.10114)](https://arxiv.org/abs/2602.10114)**
- Built upon [NVIDIA CuRobo](https://curobo.org/) and inspired by [STORM](https://github.com/NVlabs/storm).
- Supported by Ben-Gurion University of the Negev through the Agricultural, Biological and Cognitive Robotics Initiative, and the Marcus Endowment Fund for Artificial Intelligence.

## License
See [LICENSE](LICENSE) for details.
