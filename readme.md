# MR-STORM (Multi-Robot STORM) Official Repository 

<img width="1351" height="334" alt="image" src="https://github.com/user-attachments/assets/296e01e6-7869-45dd-bd94-4688dae235bc" />

## Intro
### Before we start
1. legacy names:
- "rl_for_curobo" is a legacy name. You can "pretend" it's not there. 
please ignore the legacy name, all contents of this repo are for Multi-Robot STORM (MR-STORM) a novel approach for decentralized multi-robot motion planning (particularly for manipulation tasks).

### Introduction- what is MR-STORM? 
Multi-arm MPC (Model Predictive Control) project for robotics. This repository contains a  decentralized MPC (MPPI, STORM BASED) framework for multi-arm robotic systems.
It is:
1. General; capable of solving multiple manipulation tasks in different, challenging and realistic domains.
2. Efficient; leveraging GPU for fast high control frequencies, relevant for dynamic-unpredictable domains.
3. Scalable. MR-STORM is a fully-decentralized motion planner. Meaning that it can scale properly to large number of arms, even with high number of DoFs, thanks to it's distributed nature (unlike centralized methods).   
4. Novel; offering new policy-broadcasting paradigm, along with new dead(live)lock avoidance approach, all under one decentralized approach shared accorss agents.
5. Reactive; enables real time control using STORM (https://arxiv.org/abs/2104.13542) as a "backbone", enabling it to avoid robot-robot, robot-self, and robot-environment collisions.

## Installation
### Requirements

#### System Requirements
- **OS**: Ubuntu >= 20.04
- **GPU**: NVIDIA GPU with 8+ GB VRAM (RTX 4060 or higher recommended)
- **RAM**: 32 GB
- **Other**: git, conda, git-lfs

> **Note**: We developed primarily with RTX 4060, which is below the minimum specs for some components, but works well for most use cases.

#### Supported Isaac Sim Versions
- **Isaac Sim 4.5** (Python 3.10, ldd 2.34+)
- **Isaac Sim 5.0** (Python 3.11, ldd 2.35+)

Before proceeding, check the [Isaac Sim 4.5 installation docs](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/install_python.html) and [Isaac Sim 5.0 installation docs](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/install_python.html) to determine which version is compatible with your system (primarily depends on your ldd version).

### Installation Steps

#### Step 1: Create Conda Environment

Choose a name for your environment (replace `<env_name>` with your chosen name):

```bash
# For Isaac Sim 4.5 (Python 3.10)
conda create -n <env_name> python=3.10

# OR for Isaac Sim 5.0 (Python 3.11)
conda create -n <env_name> python=3.11

# Activate the environment
conda activate <env_name>
```

#### Step 2: Install Isaac Sim

We recommend installing Isaac Sim using pip within the conda environment. Alternative installation methods (Docker, from source, etc.) are also supported, but ensure all packages are installed in the same Python environment.

##### Option A: Isaac Sim 4.5 (Python 3.10)

```bash
conda activate <env_name>
pip install isaacsim[all]==4.5.0 --extra-index-url https://pypi.nvidia.com
pip install isaacsim[extscache]==4.5.0 --extra-index-url https://pypi.nvidia.com
```

##### Option B: Isaac Sim 5.0 (Python 3.11)

```bash
conda activate <env_name>
pip install isaacsim[all,extscache]==5.0.0 --extra-index-url https://pypi.nvidia.com
```

> **Note**: For detailed installation instructions, refer to the [official Isaac Sim Python installation guide](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_python.html).

#### Step 3: Clone Repository

```bash
git clone https://github.com/RoboWorkshop/rl_for_curobo.git
cd rl_for_curobo
```

> **Note**: The repository name `rl_for_curobo` is a legacy name. This will be changed to `mpc-multi-arm` or similar in future versions.

#### Step 4: Install CuRobo

```bash
cd curobo
pip install -e . --no-build-isolation
```

If the installation fails, try:

```bash
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO=0.0.0+local pip install -e . --no-build-isolation
```

> **Note**: You don't need to clone the CuRobo repository separately - it's already included in this repository. Do not run `git clone https://github.com/NVlabs/curobo.git`.

For more information, see the [CuRobo installation documentation](https://curobo.org/get_started/1_install_instructions.html).

#### Step 5: Install rl_for_curobo Module

```bash
cd ..  # Return to rl_for_curobo root directory
pip install .
```

You should see installation logs and a success message: `Successfully installed rl_for_curobo-0.1.0`

#### Step 6: Setup Git LFS

```bash
git lfs install
git lfs pull  # Important: Pulls large files like robot meshes
```

#### Step 7: Verify Installation

Run a hello world example to verify everything is working correctly:

```bash
# TODO: Add example command here
```

## Getting Started- Run MR-STORM Examples

Once installation is complete, you can start using the multi-arm MPC framework. See the `examples/` directory for usage examples.
- If "examples" folder is missing in this branch, or from any other reason, one can run demos in the next way:
```bash
# step 1 - enter this directory
cd rl_for_curobo
# step 2- start conda env (see above)
conda activate <env_name>

# step 3- prepper config files for running
## Copy this file (combo config- in charge for a sequence  of >=1 experiments in a row), take it from:
projects_root/experiments/benchmarks/cfgs/combo_cfg.yml # and copy it to anywhere you would like (let's call assume the destination path you selected is "<combo/dst/path.yml>"


# step 4- modify the combo config file as you wish. These are the algorithms served us in the paper during benchmarking.

## alg: O means MR-STORM, O- means MR-STORM with tau=0 etc..
## robot fam- defines the robot model (arms). ur5e was used in experiments.

## robot type- first number specifies the robot count, second number (after the "_") specifies the distances between arm bases. For example, 4_05 means 4 arms, 0.5 meters distance. You can always make your urdf files and select proper names.
## task_seed- specifies the randomness seed, used for tasks (see experiments from paper).
## task_to_levels: a dict where key is the task type and value is the task level (again, see paper).

## Example of file contents to change:
# base: [projects_root/experiments/benchmarks/cfgs/meta_cfg_arms.yml]
# # alg: [SD, CC, O, SC, D, O-]
# alg: [O-] # , SD, CC, O, O-] #'O'] # ['SC'] # [O] # [O, SC, SD, CC, O-]
# robot_fam: ['ur5e'] # ['franka'] #  #[ ur5e]
# robot_type: ['4_05'] # ['4_05'] # ['4_05'] # ['2_05b'] # ['2_05'] # ['2_05', '4_04']
# # task_seed: [0,1,2] #[0,1,2]
# task_seed: [0]
# task_to_levels:
#   # manual: [1]
#   bin: [5]
#   # reach: [9]
#   # follow: [5]
#   # reach: [1,2,3,4,5]
#   # follow: [1,2,3,4,5]
#   # bin: [1,2,3,4,5]
  
# # particle: {'init_cov':[0.05,0.3, 0.5],'wta_trust':[1.0,2.0, 3.0, 5.0],'wta_weight':[50000]}



# Step 5- run the next --help flaf for more input arguments you'll need (to understand their meaning)
python /home/dan/rl_for_curobo/projects_root/experiments/core_api/dataset_collector.py --help


# Step 6- run the hello-workd experiments:
# Examples: 
#1)  default setups
python3 /home/dan/rl_for_curobo/projects_root/experiments/core_api/dataset_collector.py
#2) specify another combo config file
python3 /home/dan/rl_for_curobo/projects_root/experiments/core_api/dataset_collector.py /--combo_cfg_path "<combo/dst/path.yml>"
#3) run headless:
python3 /home/dan/rl_for_curobo/projects_root/experiments/core_api/dataset_collector.py /--combo_cfg_path "<combo/dst/path.yml>" --vis_moed headless # can also set gui or livestram if running in cluster

```
## Additional Resources
- [MR-STORM official website](https://roboworkshop.github.io/multi-robot-mpc/)
- [CuRobo Documentation](https://curobo.org/)
- [Storm Github](https://github.com/NVlabs/storm)
- [Isaac Sim Documentation](https://docs.isaacsim.omniverse.nvidia.com/)

## License

See LICENSE file for details.


