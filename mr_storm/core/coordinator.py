"""
Multi-Arm Coordinator (`MRStormCoordinator`) for running N decentralized
MR-STORM agents against any simulator (Gazebo, Isaac Sim, MuJoCo, Kinematic)
or physical multi-arm hardware.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np

from mr_storm.core.agent import (
    MRStormAgent,
    calculate_robot_sphere_count,
    resolve_repo_path,
)
from mr_storm.core.types import (
    ArmConfig,
    BroadcastPlanPacket,
    JointCommand,
    RobotJointState,
)


class MRStormCoordinator:
    """
    Coordinates N decentralized MR-STORM agents on a single machine (shared memory
    plan-board) and steps them with any `RobotBackend`.

    For multi-process / multi-machine distributed execution over ROS 2 DDS, use
    `mr_storm.ros2.distributed_arm_node.MRStormDistributedArmNode` instead.
    """

    def __init__(
        self,
        arm_configs: List[ArmConfig],
        world_cfg: Optional[object] = None,
        broadcast_full_horizon: bool = True,
    ):
        from curobo.rollout.cost.custom.arm_base.dynamic_obs_cost import DynamicObsCost
        from curobo.util_file import load_yaml
        from projects_root.projects.dynamic_obs.dynamic_obs_predictor.runtime_topics import (
            get_topics,
            init_runtime_topics,
        )

        self.arm_configs = arm_configs
        self.n_arms = len(arm_configs)
        self.broadcast_full_horizon = broadcast_full_horizon

        init_runtime_topics(n_envs=1, robots_per_env=self.n_arms)
        env_topics = get_topics().get_default_env()
        DynamicObsCost._assignment_index = 0

        robot_cfgs = []
        robot_paths = []
        particle_paths = []
        particle_cfgs = []
        sphere_splits = []
        sphere_totals = []

        for cfg in arm_configs:
            r_path = resolve_repo_path(cfg.robot_yml)
            p_path = resolve_repo_path(cfg.particle_yml)
            r_dict = load_yaml(r_path)["robot_cfg"]
            if cfg.retract_cfg is not None:
                r_dict["kinematics"]["cspace"]["retract_config"] = cfg.retract_cfg
            p_dict = load_yaml(p_path)
            b_cnt, e_cnt = calculate_robot_sphere_count(r_dict)

            robot_paths.append(r_path)
            particle_paths.append(p_path)
            robot_cfgs.append(r_dict)
            particle_cfgs.append(p_dict)
            sphere_splits.append((b_cnt, e_cnt))
            sphere_totals.append(b_cnt + e_cnt)

        # Register all robot contexts in runtime_topics before instantiating MpcSolvers
        peers_per_arm: List[List[int]] = []
        for idx, cfg in enumerate(arm_configs):
            peers = (
                cfg.override_col_With
                if cfg.override_col_With is not None
                else [j for j in range(self.n_arms) if j != idx]
            )
            peers_per_arm.append(peers)
            n_obs_spheres = sum(sphere_totals[j] for j in peers)

            ctx = env_topics[idx]
            ctx["env_id"] = 0
            ctx["robot_id"] = idx
            ctx["robot_pose"] = list(cfg.base_pose)
            ctx["n_obstacle_spheres"] = n_obs_spheres
            ctx["n_own_spheres"] = sphere_totals[idx]
            ctx["horizon"] = particle_cfgs[idx]["model"]["horizon"]
            ctx["n_rollouts"] = particle_cfgs[idx]["mppi"]["num_particles"]
            ctx["col_pred_with"] = peers
            ctx["mpc_config_paths"] = particle_paths
            ctx["robot_config_paths"] = robot_paths
            ctx["robot_sphere_counts"] = sphere_splits

        self.agents: List[MRStormAgent] = []
        for idx, cfg in enumerate(arm_configs):
            agent = MRStormAgent(
                robot_id=idx,
                base_pose=cfg.base_pose,
                robot_cfg=robot_cfgs[idx],
                particle_file_path=particle_paths[idx],
                world_cfg=world_cfg,
                peer_robot_ids=peers_per_arm[idx],
            )
            self.agents.append(agent)

        self.plan_board: Dict[int, BroadcastPlanPacket] = {}

    def set_goals_world(
        self,
        goals_W: Dict[int, Tuple[np.ndarray, np.ndarray]],
    ):
        """
        Update world-frame target poses for any subset of arms.

        Args:
            goals_W: Mapping `{arm_id: (position_xyz, quaternion_wxyz)}`
        """
        for arm_id, (pos_W, quat_W) in goals_W.items():
            self.agents[arm_id].set_goal_world(pos_W, quat_W)

    def step(
        self,
        joint_states: Dict[int, RobotJointState],
    ) -> Dict[int, JointCommand]:
        """
        Execute one decentralized multi-arm MPC step:
        1. Each arm broadcasts its predicted horizon collision spheres + goal error.
        2. Each arm updates its `DynamicObsCollPredictor` and priority weights from peers.
        3. Each arm solves its local GPU MPPI optimization and returns a `JointCommand`.
        """
        # 1. Broadcast predicted horizon collision spheres from current policy means
        for idx, agent in enumerate(self.agents):
            js = joint_states[idx]
            self.plan_board[idx] = agent.build_broadcast_plan(
                js, naive_static_broadcast=(not self.broadcast_full_horizon)
            )

        # 2. Update each agent's collision predictor with peer plans & priority errors
        for idx, agent in enumerate(self.agents):
            agent.update_from_peer_plans(self.plan_board)

        # 3. Solve local MPPI step for each arm
        commands: Dict[int, JointCommand] = {}
        for idx, agent in enumerate(self.agents):
            commands[idx] = agent.step(joint_states[idx])

        return commands
