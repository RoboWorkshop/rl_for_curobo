#!/usr/bin/env python3
#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#


try:
    # Third Party
    import isaacsim
except ImportError:
    pass


# Third Party
import torch
import time
a = torch.zeros(4, device="cuda:0")

# Standard Library
import argparse

## import curobo:

parser = argparse.ArgumentParser()

parser.add_argument(
    "--headless_mode",
    type=str,
    default=None,
    help="To run headless, use one of [native, websocket], webrtc might not work.",
)
parser.add_argument(
    "--visualize_spheres",
    action="store_true",
    help="When True, visualizes robot spheres",
    default=False,
)

parser.add_argument("--robot", type=str, default="g1/left_arm.yml", help="robot configuration to load")
args = parser.parse_args()

###########################################################

# Third Party
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": args.headless_mode is not None,
        "width": "1920",
        "height": "1080",
    }
)

# Third Party
# Enable the layers and stage windows in the UI
# Standard Library
import os

# Third Party
import carb
import numpy as np
from projects_root.examples.helper import add_robot_to_scene
from omni.isaac.core import World
from omni.isaac.core.objects import cuboid, VisualSphere
from omni.isaac.core.utils.types import ArticulationAction

# CuRobo
from curobo.util.logger import setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
# Create pose and update goal
from curobo.types.math import Pose

############################################################
# Global coordinate frame configuration
############################################################

# World frame: fixed reference frame (identity transform)
WORLD_FRAME = {
    'position': np.array([0.0, 0.0, 0.0]),
    'orientation': np.array([1.0, 0.0, 0.0, 0.0])  # [w, x, y, z] quaternion
}

# Base link frame: will be set from robot configuration
BASE_LINK_FRAME = {
    'name': None,  # Will be set from robot config
    'robot_instance': None  # Will store robot instance for pose queries
}

def set_base_link_frame(robot_config: dict, robot_instance):
    """Set the base link frame from robot configuration and instance.
    
    Args:
        robot_config: Robot configuration dictionary
        robot_instance: Isaac Sim robot instance for pose queries
    """
    global BASE_LINK_FRAME
    BASE_LINK_FRAME['name'] = robot_config["kinematics"]["base_link"]
    BASE_LINK_FRAME['robot_instance'] = robot_instance
    print(f"Base link frame set to: {BASE_LINK_FRAME['name']}")

def transform_pose_from_to(position, orientation, from_frame='world_frame', to_frame='base_link_frame'):
    """Transform pose between coordinate frames.
    
    Args:
        position: Position as numpy array [x, y, z]
        orientation: Orientation as quaternion [w, x, y, z] (Isaac Sim format)
        from_frame: Source coordinate frame ('world_frame' or 'base_link_frame')
        to_frame: Target coordinate frame ('world_frame' or 'base_link_frame')
        
    Returns:
        tuple: (transformed_position, transformed_orientation)
    """
    
    if from_frame == to_frame:
        return position, orientation
    
    # Get robot instance
    if BASE_LINK_FRAME['robot_instance'] is None:
        raise ValueError("Base link frame not initialized. Call set_base_link_frame() first.")
    
    robot = BASE_LINK_FRAME['robot_instance']
    
    # Import scipy rotation for coordinate transformations
    from scipy.spatial.transform import Rotation as R

    # Get robot's current world pose (this is the base_link world pose)
    robot_world_position, robot_world_orientation = robot.get_world_pose()
    
    def isaac_to_scipy_quat(quat):
        return np.array([quat[1], quat[2], quat[3], quat[0]])
    
    def scipy_to_isaac_quat(quat):
        return np.array([quat[3], quat[0], quat[1], quat[2]])
    
    if from_frame == 'world_frame' and to_frame == 'base_link_frame':
        # Transform from world to robot base frame
        
        # Position transformation
        robot_rotation = R.from_quat(isaac_to_scipy_quat(robot_world_orientation))
        world_to_robot_translation = position - robot_world_position
        transformed_position = robot_rotation.inv().apply(world_to_robot_translation)
        
        # Orientation transformation
        target_rotation = R.from_quat(isaac_to_scipy_quat(orientation))
        relative_rotation = robot_rotation.inv() * target_rotation
        transformed_orientation = scipy_to_isaac_quat(relative_rotation.as_quat())
        
        return transformed_position, transformed_orientation
        
    elif from_frame == 'base_link_frame' and to_frame == 'world_frame':
        # Transform from robot base frame to world
        
        # Position transformation
        robot_rotation = R.from_quat(isaac_to_scipy_quat(robot_world_orientation))
        robot_to_world_translation = robot_rotation.apply(position)
        transformed_position = robot_to_world_translation + robot_world_position
        
        # Orientation transformation
        relative_rotation = R.from_quat(isaac_to_scipy_quat(orientation))
        world_rotation = R.from_quat(isaac_to_scipy_quat(robot_world_orientation))
        target_rotation = world_rotation * relative_rotation
        transformed_orientation = scipy_to_isaac_quat(target_rotation.as_quat())
        
        return transformed_position, transformed_orientation
        
    else:
        raise ValueError(f"Unsupported frame transformation: {from_frame} -> {to_frame}")

def update_target_goal(cube_position, cube_orientation, goal_buffer, mpc, tensor_args):
    """Update MPC goal with coordinate frame transformation.
    
    Args:
        cube_position: Target position in world frame
        cube_orientation: Target orientation in world frame  
        goal_buffer: MPC goal buffer to update
        mpc: MPC solver instance
        tensor_args: Tensor device arguments
    """
    # Transform target from world frame to robot base frame
    ee_translation_goal, ee_orientation_goal = transform_pose_from_to(
        cube_position, cube_orientation, 
        from_frame='world_frame', 
        to_frame='base_link_frame'
    )
    

    ik_goal = Pose(
        position=tensor_args.to_device(ee_translation_goal),
        quaternion=tensor_args.to_device(ee_orientation_goal),
    )
    goal_buffer.goal_pose.copy_(ik_goal)
    mpc.update_goal(goal_buffer)
    
    # Debug output
    print(f"Target world pos: {cube_position}")
    print(f"Target robot-relative pos: {ee_translation_goal}")
    
    return ik_goal

############################################################


########### OV #################;;;;;


###########
EXT_DIR = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__))))
DATA_DIR = os.path.join(EXT_DIR, "data")
########### frame prim #################;;;;;


# Standard Library
from typing import Optional

# Third Party
from projects_root.examples.helper import add_extensions, add_robot_to_scene

# CuRobo
# from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.rollout.rollout_base import Goal
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.types.state import JointState
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig

############################################################


def draw_points(rollouts: torch.Tensor):
    if rollouts is None:
        return
    # Standard Library
    import random

    # Third Party
    try:
        from omni.isaac.debug_draw import _debug_draw
    except ImportError:
        from isaacsim.util.debug_draw import _debug_draw
    draw = _debug_draw.acquire_debug_draw_interface()
    N = 100
    # if draw.get_num_points() > 0:
    draw.clear_points()
    cpu_rollouts = rollouts.cpu().numpy()
    b, h, _ = cpu_rollouts.shape
    point_list = []
    colors = []
    
    # Transform rollout points from robot base frame to world frame for visualization
    for i in range(b):
        for j in range(h):
            # Get point in robot base frame
            base_frame_position = cpu_rollouts[i, j, :3]  # [x, y, z]
            base_frame_orientation = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion [w, x, y, z]
            
            try:
                # Transform to world frame for visualization
                world_position, _ = transform_pose_from_to(
                    base_frame_position, base_frame_orientation,
                    from_frame='base_link_frame', 
                    to_frame='world_frame'
                )
                point_list.append((world_position[0], world_position[1], world_position[2]))
            except (ValueError, Exception):
                # Fallback: use original points if transformation fails
                point_list.append((cpu_rollouts[i, j, 0], cpu_rollouts[i, j, 1], cpu_rollouts[i, j, 2]))
        
        # Add colors for this batch
        colors += [(1.0 - (i + 1.0 / b), 0.3 * (i + 1.0 / b), 0.0, 0.1) for _ in range(h)]
    
    sizes = [10.0 for _ in range(b * h)]
    draw.draw_points(point_list, colors, sizes)


def main():
    # assuming obstacles are in objects_path:
    my_world = World(stage_units_in_meters=1.0)
    stage = my_world.stage

    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")
    # my_world.stage.SetDefaultPrim(my_world.stage.GetPrimAtPath("/World"))
    stage = my_world.stage
    my_world.scene.add_default_ground_plane()

    # stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

    # Make a target to follow
    target = cuboid.VisualCuboid(
        "/World/target",
        position=np.array([0.5, 0, 0.5]),
        orientation=np.array([0, 1, 0, 0]),
        color=np.array([1.0, 0, 0]),
        size=0.05,
    )

    setup_curobo_logger("warn")
    past_pose = None
    n_obstacle_cuboids = 30
    n_obstacle_mesh = 10

    # warmup curobo instance
    usd_help = UsdHelper()
    target_pose = None

    tensor_args = TensorDeviceType()

    robot_cfg = load_yaml(join_path(get_robot_configs_path(), args.robot))["robot_cfg"]

    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]
    # robot_cfg["kinematics"]["collision_sphere_buffer"] += 0.02

    # Use URDF import path (load_from_usd=False) to avoid prim-composition timing issues
    robot, robot_prim_path = add_robot_to_scene(
        robot_cfg,
        my_world,
        load_from_usd=False,
        position=np.array([0.0, 0.0, 0.7])
    )

    # Initialize coordinate frame system
    set_base_link_frame(robot_cfg, robot)

    articulation_controller = robot.get_articulation_controller()

    world_cfg_table = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    )
    world_cfg_table.cuboid[0].pose[2] -= 0.04
    world_cfg1 = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    ).get_mesh_world()
    world_cfg1.mesh[0].name += "_mesh"
    world_cfg1.mesh[0].pose[2] = -10.5

    world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid, mesh=world_cfg1.mesh)

    init_curobo = False

    tensor_args = TensorDeviceType()

    robot_cfg = load_yaml(join_path(get_robot_configs_path(), args.robot))["robot_cfg"]

    world_cfg_table = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    )
    world_cfg1 = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    ).get_mesh_world()
    world_cfg1.mesh[0].pose[2] = -10.0

    world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid, mesh=world_cfg1.mesh)
    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]

    default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]

    mpc_config = MpcSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        use_cuda_graph=True,
        use_cuda_graph_metrics=True,
        use_cuda_graph_full_step=False,
        self_collision_check=True,
        collision_checker_type=CollisionCheckerType.MESH,
        collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
        use_mppi=True,
        use_lbfgs=False,
        use_es=False,
        store_rollouts=True,
        step_dt=0.02,
        override_particle_file="projects_root/real_world/particle_mpc.yml",
    )

    mpc = MpcSolver(mpc_config)

    retract_cfg = mpc.rollout_fn.dynamics_model.retract_config.clone().unsqueeze(0)
    joint_names = mpc.rollout_fn.joint_names

    state = mpc.rollout_fn.compute_kinematics(
        JointState.from_position(retract_cfg, joint_names=joint_names)
    )
    current_state = JointState.from_position(retract_cfg, joint_names=joint_names)
    retract_pose = Pose(state.ee_pos_seq, quaternion=state.ee_quat_seq)
    goal = Goal(
        current_state=current_state,
        goal_state=JointState.from_position(retract_cfg, joint_names=joint_names),
        goal_pose=retract_pose,
    )

    goal_buffer = mpc.setup_solve_single(goal, 1)
    mpc.update_goal(goal_buffer)
    mpc_result = mpc.step(current_state, max_attempts=2)
   
    usd_help.load_stage(my_world.stage)
    init_world = False
    cmd_state_full = None
    step = 0
    add_extensions(simulation_app, args.headless_mode)
    spheres = None  # For collision sphere visualization
    while simulation_app.is_running():
        if not init_world:
            for _ in range(10):
                my_world.step(render=True)
            init_world = True
        draw_points(mpc.get_visual_rollouts())

        my_world.step(render=True)
        if not my_world.is_playing():
            continue

        step_index = my_world.current_time_step_index

        if step_index <= 10:
            # my_world.reset()
            robot._articulation_view.initialize()

            idx_list = [robot.get_dof_index(x) for x in j_names]
            robot.set_joint_positions(default_config, idx_list)

            robot._articulation_view.set_max_efforts(
                values=np.array([5000 for i in range(len(idx_list))]), joint_indices=idx_list
            )

        if not init_curobo:
            init_curobo = True
        step += 1
        step_index = step
        if step_index % 1000 == 0:
            print("Updating world")
            obstacles = usd_help.get_obstacles_from_stage(
                only_paths=["/World"],
                ignore_substring=[
                    robot_prim_path,
                    "/World/target",
                    "/World/defaultGroundPlane",
                    "/curobo",
                ],
                reference_prim_path=robot_prim_path,
            )
            obstacles.add_obstacle(world_cfg_table.cuboid[0])
            mpc.world_coll_checker.load_collision_model(obstacles)

        # position and orientation of target virtual cube:
        cube_position, cube_orientation = target.get_world_pose()

        if past_pose is None:
            past_pose = cube_position + 1.0

        if np.linalg.norm(cube_position - past_pose) > 1e-3:
            # Use the clean coordinate transformation function
            update_target_goal(cube_position, cube_orientation, goal_buffer, mpc, tensor_args)
            past_pose = cube_position

        # if not changed don't call curobo:

        # get robot current state:
        sim_js = robot.get_joints_state()
        if sim_js is None:
            print("sim_js is None")
            continue
        js_names = robot.dof_names
        sim_js_names = robot.dof_names

        cu_js = JointState(
            position=tensor_args.to_device(sim_js.positions),
            velocity=tensor_args.to_device(sim_js.velocities) * 0.0,
            acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
            joint_names=sim_js_names,
        )
        cu_js = cu_js.get_ordered_joint_state(mpc.rollout_fn.joint_names)
        if cmd_state_full is None:
            current_state.copy_(cu_js)
        else:
            current_state_partial = cmd_state_full.get_ordered_joint_state(
                mpc.rollout_fn.joint_names
            )
            current_state.copy_(current_state_partial)
            current_state.joint_names = current_state_partial.joint_names
            # current_state = current_state.get_ordered_joint_state(mpc.rollout_fn.joint_names)
        common_js_names = []
        current_state.copy_(cu_js)

        # --- Collision sphere visualization ---
        if args.visualize_spheres and step_index % 2 == 0:
            sph_list = mpc.kinematics.get_robot_as_spheres(cu_js.position)
            # Get base link world pose (position, orientation as quaternion)
            base_pos, base_quat = robot.get_world_pose()  # base_link is "pelvis"
            from scipy.spatial.transform import Rotation as R
            def isaac_to_scipy_quat(quat):
                return np.array([quat[1], quat[2], quat[3], quat[0]])
            base_rot = R.from_quat(isaac_to_scipy_quat(base_quat))
            if spheres is None:
                spheres = []
                for si, s in enumerate(sph_list[0]):
                    # Transform sphere position from base_link to world
                    sphere_pos_world = base_rot.apply(s.position) + base_pos
                    sp = VisualSphere(
                        prim_path="/curobo/robot_sphere_" + str(si),
                        position=np.ravel(sphere_pos_world),
                        radius=float(s.radius),
                        color=np.array([0, 0.8, 0.2]),
                    )
                    spheres.append(sp)
            else:
                for si, s in enumerate(sph_list[0]):
                    if not np.isnan(s.position[0]):
                        sphere_pos_world = base_rot.apply(s.position) + base_pos
                        spheres[si].set_world_pose(position=np.ravel(sphere_pos_world))
                        spheres[si].set_radius(float(s.radius))
        mpc_time = time.time()
        mpc_result = mpc.step(current_state, max_attempts=2)
        mpc_time = time.time() - mpc_time
        print(f"MPC time: {mpc_time}")
        # ik_result = ik_solver.solve_single(ik_goal, cu_js.position.view(1,-1), cu_js.position.view(1,1,-1))

        succ = True  # ik_result.success.item()
        cmd_state_full = mpc_result.js_action
        common_js_names = []
        idx_list = []
        for x in sim_js_names:
            if x in cmd_state_full.joint_names:
                idx_list.append(robot.get_dof_index(x))
                common_js_names.append(x)

        cmd_state = cmd_state_full.get_ordered_joint_state(common_js_names)
        cmd_state_full = cmd_state

        art_action = ArticulationAction(
            cmd_state.position.view(-1).cpu().numpy(),
            # cmd_state.velocity.cpu().numpy(),
            joint_indices=idx_list,
        )
        # positions_goal = articulation_action.joint_positions
        if step_index % 1000 == 0:
            print(mpc_result.metrics.feasible.item(), mpc_result.metrics.pose_error.item())

        if succ:
            # set desired joint angles obtained from IK:
            for _ in range(1):
                articulation_controller.apply_action(art_action)

        else:
            carb.log_warn("No action is being taken.")


############################################################

if __name__ == "__main__":
    main()
    simulation_app.close()
