# MR-STORM: Scalable Multi-Arm Control By Distributed MPC

[![IEEE/RSJ IROS 2026](https://img.shields.io/badge/IEEE%2FRSJ-IROS%202026-00629B.svg)](https://roboworkshop.github.io/multi-robot-mpc/)
[![arXiv](https://img.shields.io/badge/arXiv-2602.10114-b31b1b.svg)](https://arxiv.org/abs/2602.10114)
[![Project Page](https://img.shields.io/badge/Project-Website-2ea44f.svg)](https://roboworkshop.github.io/multi-robot-mpc/)
[![Simulators](https://img.shields.io/badge/Simulators-Isaac%20Sim%20%7C%20Gazebo%20%7C%20Kinematic-orange.svg)](#unified-entry-point-mainpy--agnostic-layer)
[![ROS](https://img.shields.io/badge/ROS-Sync%20%26%20Async%20Decentralized-22314E.svg)](#execution-modes-synchronous-decentralized-vs-asynchronous-decentralized)

<p align="center">
  <img width="100%" alt="MR-STORM Banner" src="https://github.com/user-attachments/assets/296e01e6-7869-45dd-bd94-4688dae235bc" />
</p>

> **Official Repository for the IEEE/RSJ IROS 2026 Paper:**  
> **[MR-STORM: Scalable Multi-Arm Control By Distributed MPC](https://arxiv.org/abs/2602.10114)**  
> [Dan Evron](mailto:evrond@post.bgu.ac.il), [Elias Goldsztejn](mailto:eliasgol@post.bgu.ac.il), [Dan R. Suissa](mailto:danrouve@post.bgu.ac.il), and [Ronen I. Brafman](mailto:brafman@bgu.ac.il)  
> *Foundations of AI Institute, Stein Faculty of Computer and Information Science, Ben-Gurion University of the Negev*

---

## Overview

Centralized motion planning algorithms struggle to scale in multi-arm manipulation, where tight shared workspaces and inter-arm coupling cause the joint state space to grow exponentially with the number of arms.

**MR-STORM** is a decentralized, GPU-accelerated Model Predictive Path Integral (MPPI) framework building on [STORM](https://arxiv.org/abs/2104.13542) and [CuRobo](https://curobo.org/) that controls multiple robotic arms in real time (30–100+ Hz):

1. **Decentralized & Scalable:** Each robotic arm runs its own local GPU MPPI trajectory optimizer (`CuAgent` / `MpcPlanner`), scaling linearly in per-agent computation to 4+ high-DoF manipulators in tight workspaces.
2. **Horizon Plan-Sharing:** At every control cycle, each arm integrates its MPPI mean acceleration policy (`get_estimated_plan`) over the horizon $H$ and shares its predicted task-space collision spheres $\{(p_{t,s}, r_s)\}_{t=1..H}$ in the world frame (`plans_board`).
3. **Dynamic Priority Conflict Resolution ($\tau$):** To prevent livelocks and deadlocks in symmetric bottlenecks, each arm weights inter-arm collision costs (`update_col_pred`) by relative goal proximity (`mean_goal_err`):
   $$d_{\text{eff}}(i, j) = d(i, j) \cdot \left(\frac{e_{\text{goal}}(j)}{e_{\text{goal}}(i)}\right)^{\tau}$$
   where $\tau \ge 0$ (`wta_trust`) smoothly prioritizes arms closer to completing their current task.
4. **Unified `main.py` & Simulator/Hardware-Agnostic Layer (`MRStormAgnosticLayer`):** Stays 100% loyal to the original `benchmark_sim.py` classes (`CuAgent`, `MpcPlanner`, `CumotionPlanner`, `PlanPubSub`, `WorldModelWrapper`) while taking observations (`RobotObservation`) and returning embodiment actions (`EmbodimentAction`) for **any simulator** (Isaac Sim, Gazebo, MuJoCo, Kinematic) or **any real hardware** (UR5e/UR10e, Franka, Unitree G1, etc.).

---

## Repository Structure

```text
├── main.py                         # Unified simulator- & hardware-agnostic entry point (sync & async modes)
├── mr_storm/                       # Dedicated agnostic layer & loyal core classes
│   ├── core/
│   │   ├── planners.py             # Loyal extraction of CuPlanner, MpcPlanner, CumotionPlanner, CuAgent, PlanPubSub
│   │   ├── agnostic_layer.py       # MRStormAgnosticLayer: takes RobotObservation -> returns EmbodimentAction
│   │   ├── agent.py                # Standalone single-arm wrapper
│   │   └── coordinator.py          # Multi-arm coordinator
│   ├── comm/
│   │   └── plan_comm.py            # LocalPlanBoardComm (sync/shared) & ROSPlanBoardComm (async over ROS 2 / ROS 1)
│   ├── backends/
│   │   ├── env_adapters.py         # KinematicEnvAdapter, GazeboROSAdapter, RealHardwareEnvAdapter
│   │   ├── gazebo_backend.py       # Gazebo ros2_control configuration utilities
│   │   └── real_robot_backend.py   # SafetySupervisor + URRTDEHardwareDriver + CallbackHardwareDriver
│   └── ros2/                       # ROS 2 launch files and multi-arm fleet configs
├── examples/                       # Quickstart scripts for Gazebo, Real Robot, ROS 2, and custom simulators
├── curobo/                         # Bundled CuRobo fork with custom DynamicObsCost GPU rollout kernels
└── projects_root/                  # IROS 2026 Isaac Sim benchmark suite, configs, and analysis tools
    ├── experiments/core_api/       # dataset_collector.py, benchmark_sim.py, and meta_conf_variants/*.yml
    └── projects/dynamic_obs/       # DynamicObsCollPredictor & runtime_topics
```

---

## Execution Modes: `synchronous-decentralized` vs `asynchronous-decentralized`

Both modes are supported in **both simulation and real robot hardware** via [`main.py`](main.py) and [`MRStormAgnosticLayer`](mr_storm/core/agnostic_layer.py):

| Mode | CLI Flag | Inter-Robot Communication | Execution Flow |
| :--- | :---: | :---: | :--- |
| **1. Synchronous-Decentralized** | `--mode sync` | `LocalPlanBoardComm` (in-memory `plans_board` & `mean_goal_err`, or `--comm ros`) | Exact execution flow of `benchmark_sim.py`: robots compute paths decentrally (`get_estimated_plan` $\to$ `plans_board` $\to$ `update_col_pred` $\to$ `yield_action`), planning one after the next in a pseudo-distributed sequential loop. |
| **2. Asynchronous-Decentralized** | `--mode async` | `ROSPlanBoardComm` (`--comm ros`, publishes `/mr_storm/robot_<i>/plan` & `/mr_storm/robot_<i>/goal_error` over ROS 2 / ROS 1) | Each robot runs its own independent control loop (in separate OS processes via `--robot_id <i>` or parallel async threads) without waiting for other robots, exchanging horizon sphere plans and goal errors over ROS. |

---

## Unified Entry Point (`main.py`) & Agnostic Layer

### 1. Run Synchronous-Decentralized (Pseudo-Distributed Sequential Planning)

```bash
# Pure-GPU kinematic backend (runs out of the box without Isaac Sim or Gazebo):
python3 main.py --mode sync --backend kinematic \
  --cfg projects_root/experiments/core_api/meta_conf_variants/meta_cfg_arms_50steps.yml

# Gazebo (ros2_control):
python3 main.py --mode sync --backend gazebo \
  --cfg projects_root/experiments/core_api/meta_conf_variants/meta_cfg_arms.yml

# Real Robot Hardware (with SafetySupervisor & UR RTDE IPs):
python3 main.py --mode sync --backend real --ur_ips 192.168.1.101,192.168.1.102

# NVIDIA Isaac Sim (full stage & physics):
python3 main.py --mode sync --backend isaac --vis gui
```

### 2. Run Asynchronous-Decentralized (Communicating Over ROS)

```bash
# Option A: Run each robot in its own independent OS process / machine over ROS:
python3 main.py --mode async --comm ros --backend real --robot_id 0 --ur_ips 192.168.1.101 &
python3 main.py --mode async --comm ros --backend real --robot_id 1 --ur_ips 192.168.1.102 &

# Option B: Run all robots asynchronously in parallel communicating over ROS in Gazebo:
python3 main.py --mode async --comm ros --backend gazebo \
  --cfg projects_root/experiments/core_api/meta_conf_variants/meta_cfg_arms.yml
```

### 3. Using `MRStormAgnosticLayer` Directly in Your Own Python Loop

```python
from curobo.util_file import load_yaml
from mr_storm import MRStormAgnosticLayer, RobotObservation

meta_cfg = load_yaml("projects_root/experiments/core_api/meta_conf_variants/meta_cfg_arms_50steps.yml")

# Create the agnostic layer ('synchronous-decentralized' or 'asynchronous-decentralized')
layer = MRStormAgnosticLayer(meta_cfg, mode="synchronous-decentralized", comm_type="local")

# Pass simulation or real observations -> receive EmbodimentAction per robot
observations = layer.get_retract_observations()  # or populate RobotObservation from your sim/robot
for t in range(100):
    actions = layer.step_synchronous_decentralized(observations, t)
    for robot_idx, act in actions.items():
        # act.positions, act.velocities, act.accelerations, act.joint_names, act.cu_action
        pass
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

If you want to run the **Isaac Sim** benchmark suite (Gazebo, Real Robot, and Kinematic backends do **not** require Isaac Sim):

```bash
# Option A: Isaac Sim 4.5 (Python 3.10, glibc/ldd >= 2.34)
pip install isaacsim[all,extscache]==4.5.0 --extra-index-url https://pypi.nvidia.com

# Option B: Isaac Sim 5.0 (Python 3.11, glibc/ldd >= 2.35)
pip install isaacsim[all,extscache]==5.0.0 --extra-index-url https://pypi.nvidia.com
```

---

## Reproducing the IROS 2026 Paper Benchmarks (Isaac Sim)

All simulation benchmarks from the paper are configured via [`projects_root/experiments/benchmarks/cfgs/combo_cfg.yml`](projects_root/experiments/benchmarks/cfgs/combo_cfg.yml) and executed through [`dataset_collector.py`](projects_root/experiments/core_api/dataset_collector.py).

| Code | Algorithm | Plan Broadcasting | Priority ($\tau$) | Description |
| :---: | :--- | :---: | :---: | :--- |
| **`O`** | **MR-STORM (Ours)** | Full Horizon ($H$) | $\tau > 0$ (`pose_wta`) | Decentralized GPU MPPI with horizon plan-sharing & dynamic priority |
| **`O-`** | **MR-STORM ($\tau=0$)** | Full Horizon ($H$) | $\tau = 0$ (Equal weight) | Ablation without priority conflict resolution |
| **`SD`** | **STORM Decentralized** | Static Current Pose | None | Decentralized MPPI treating peers as static obstacles at their current pose |
| **`SC`** | **STORM Centralized** | N/A (Joint Space) | N/A | Single centralized MPPI planning over the concatenated DoFs of all arms |
| **`CC`** | **CuRobo Centralized** | N/A (Joint Space) | N/A | Centralized CuMotion graph + trajectory optimization baseline |
| **`D`** | **dRRT\*** | N/A | N/A | Sampling-based multi-robot RRT* baseline |

```bash
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

## License
See [LICENSE](LICENSE) for details.
