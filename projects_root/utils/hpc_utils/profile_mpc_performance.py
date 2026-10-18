#!/usr/bin/env python3

"""
Performance profiling script to analyze MPC client-server communication bottlenecks.
"""

import time
import pickle
import sys
import argparse
from typing import Dict, Any
import zmq
import torch

import os
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(_REPO_ROOT, 'projects_root'))
sys.path.append(os.path.join(_REPO_ROOT, 'curobo', 'src'))

from curobo.types.state import JointState
from curobo.types.base import TensorDeviceType
from curobo.wrap.reacher.mpc import MpcSolverConfig
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from utils.hpc_utils.mpc_solver_api import MpcSolverAPI


def profile_serialization(data, name="data"):
    """Profile pickle serialization performance."""
    print(f"\n=== Profiling {name} Serialization ===")
    
    # Test serialization
    start_time = time.time()
    serialized = pickle.dumps(data)
    serialize_time = time.time() - start_time
    
    # Test deserialization
    start_time = time.time()
    deserialized = pickle.loads(serialized)
    deserialize_time = time.time() - start_time
    
    print(f"Serialize time: {serialize_time*1000:.2f}ms")
    print(f"Deserialize time: {deserialize_time*1000:.2f}ms")
    print(f"Total time: {(serialize_time + deserialize_time)*1000:.2f}ms")
    print(f"Data size: {len(serialized)/1024:.1f} KB")
    
    return serialize_time + deserialize_time, len(serialized)


def create_test_joint_state():
    """Create a test JointState for profiling."""
    tensor_args = TensorDeviceType()
    
    # Create typical joint state
    joint_state = JointState(
        position=torch.randn(1, 7, **tensor_args.as_torch_dict()),
        velocity=torch.randn(1, 7, **tensor_args.as_torch_dict()),
        acceleration=torch.randn(1, 7, **tensor_args.as_torch_dict()),
        jerk=torch.randn(1, 7, **tensor_args.as_torch_dict()),
        joint_names=['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 
                    'panda_joint5', 'panda_joint6', 'panda_joint7']
    )
    
    return joint_state


def profile_network_latency(server_ip: str, server_port: int):
    """Profile basic network latency."""
    print(f"\n=== Profiling Network Latency to {server_ip}:{server_port} ===")
    
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://{server_ip}:{server_port}")
    socket.setsockopt(zmq.LINGER, 0)
    
    # Simple ping test
    ping_times = []
    for i in range(10):
        start_time = time.time()
        
        # Send simple request
        request = {"type": "ping", "data": "hello"}
        serialized_request = pickle.dumps(request)
        socket.send(serialized_request)
        
        # Receive response
        try:
            serialized_response = socket.recv()
            response = pickle.loads(serialized_response)
            
            ping_time = time.time() - start_time
            ping_times.append(ping_time)
            print(f"Ping {i+1}: {ping_time*1000:.2f}ms")
            
        except Exception as e:
            print(f"Ping {i+1} failed: {e}")
    
    socket.close()
    context.term()
    
    if ping_times:
        avg_ping = sum(ping_times) / len(ping_times)
        print(f"Average ping: {avg_ping*1000:.2f}ms")
        return avg_ping
    return 0


def profile_mpc_step(server_ip: str, server_port: int):
    """Profile a complete MPC step operation."""
    print(f"\n=== Profiling Complete MPC Step ===")
    
    # Load configuration
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
    world_cfg = load_yaml(join_path(get_world_configs_path(), "collision_base.yml"))
    
    config_params = {
        'robot_cfg': robot_cfg['robot_cfg'],
        'world_cfg': world_cfg,
        'tensor_args': TensorDeviceType(),
        'use_cuda_graph': True,
        'use_cuda_graph_metrics': True,
        'use_cuda_graph_full_step': False,
        'self_collision_check': True,
        'use_mppi': True,
        'use_lbfgs': False,
        'store_rollouts': True,
        'compute_metrics': True,
        'sync_cuda_time': True,
        'project_pose_to_goal_frame': True,
    }
    
    # Profile config serialization
    profile_serialization(config_params, "MPC Config")
    
    # Create test data
    current_state = create_test_joint_state()
    profile_serialization(current_state, "JointState")
    
    # Connect to server
    print(f"\nConnecting to MPC server...")
    start_time = time.time()
    mpc_api = MpcSolverAPI(server_ip, server_port, config_params)
    init_time = time.time() - start_time
    print(f"MPC initialization time: {init_time*1000:.2f}ms")
    
    # Profile multiple MPC steps
    step_times = []
    for i in range(5):
        print(f"\nMPC Step {i+1}:")
        
        # Detailed timing breakdown
        total_start = time.time()
        
        # Time the step call
        step_start = time.time()
        result = mpc_api.step(current_state, max_attempts=2)
        step_time = time.time() - step_start
        
        total_time = time.time() - total_start
        
        step_times.append(step_time)
        print(f"  Step time: {step_time*1000:.2f}ms")
        print(f"  Total time: {total_time*1000:.2f}ms")
        
        if result is not None:
            # Profile result serialization
            profile_serialization(result, f"Step {i+1} Result")
    
    if step_times:
        avg_step_time = sum(step_times) / len(step_times)
        print(f"\nAverage MPC step time: {avg_step_time*1000:.2f}ms")
        return avg_step_time
    
    return 0


def profile_tensor_operations():
    """Profile tensor serialization specifically."""
    print(f"\n=== Profiling Tensor Operations ===")
    
    tensor_args = TensorDeviceType()
    
    # Test different tensor sizes
    sizes = [
        (1, 7),      # Single joint state
        (1, 32, 7),  # Trajectory
        (100, 32, 7) # Batch of trajectories
    ]
    
    for size in sizes:
        print(f"\nTensor size {size}:")
        tensor = torch.randn(*size, **tensor_args.as_torch_dict())
        
        # Profile raw tensor
        profile_serialization(tensor, f"Tensor {size}")
        
        # Profile tensor in JointState
        if len(size) == 2:  # Only for compatible sizes
            joint_state = JointState(
                position=tensor,
                velocity=tensor,
                acceleration=tensor,
                jerk=tensor,
                joint_names=['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 
                            'panda_joint5', 'panda_joint6', 'panda_joint7']
            )
            profile_serialization(joint_state, f"JointState {size}")


def main():
    parser = argparse.ArgumentParser(description="Profile MPC Performance")
    parser.add_argument("--server_ip", type=str, default="localhost", help="Server IP")
    parser.add_argument("--server_port", type=int, default=10051, help="Server port")
    parser.add_argument("--skip_server", action="store_true", help="Skip server-dependent tests")
    
    args = parser.parse_args()
    
    print("=== MPC Performance Profiling ===")
    
    # Profile tensor operations (local)
    profile_tensor_operations()
    
    if not args.skip_server:
        try:
            # Profile network latency
            profile_network_latency(args.server_ip, args.server_port)
            
            # Profile complete MPC step
            profile_mpc_step(args.server_ip, args.server_port)
            
        except Exception as e:
            print(f"Server profiling failed: {e}")
            print("Make sure the MPC server is running:")
            print(f"  python mpc_server.py --port {args.server_port}")
    
    print("\n=== Profiling Complete ===")


if __name__ == "__main__":
    main() 