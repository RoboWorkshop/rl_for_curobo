from contextlib import contextmanager
from typing import Dict, Any, Optional

# Global configuration dictionary - simple and straightforward
_config = {
    'env': {0: {'robots': {0: {}}}},
    'robot_base_poses': {},  # Store robot base poses here
    # 'num_particles_rollout_full': 100,
    # 'horizon_rollout_full': 50,
    # 'prediction_frequency': 10.0,
    # 'safety_margin': 0.1,
}

def get(key, default=None):
    """Get a configuration value."""
    return _config.get(key, default)

def set(key, value, env=0):
    """Set a configuration value."""
    _config[key] = value

def set_robot_base_pose(robot_id, base_pose):
    """Set robot base pose for a specific robot."""
    _config['robot_base_poses'][robot_id] = base_pose
    print(f"[DEBUG] Set robot {robot_id} base pose: {base_pose}")

def get_robot_base_pose(robot_id, default=None):
    """Get robot base pose for a specific robot."""
    pose = _config['robot_base_poses'].get(robot_id, default)
    print(f"[DEBUG] Get robot {robot_id} base pose: {pose}")
    return pose

def update(**kwargs):
    """Update multiple configuration values at once."""
    _config.update(kwargs)

def get_all():
    """Get all configuration values."""
    return _config.copy()

def reset():
    """Reset to default values."""
    global _config
    _config = {
        'env': {0: {'robots': {0: {}}}},
        'robot_base_poses': {},
    }

@contextmanager
def temporary_config(**kwargs):
    """Context manager for temporary configuration changes.
    
    Usage:
        with temporary_config(num_particles_rollout_full=200, horizon_rollout_full=75):
            # Code that uses the temporary config
            particles = get('num_particles_rollout_full')
            horizon = get('horizon_rollout_full')
        # Config is automatically restored
    """
    # Store original values
    original_values = {}
    for key in kwargs:
        original_values[key] = get(key)
    
    # Set temporary values
    try:
        update(**kwargs)
        yield
    finally:
        # Restore original values
        update(**original_values)

@contextmanager
def config_from_dict(config_dict: Dict[str, Any]):
    """Context manager to temporarily use a different configuration.
    
    Usage:
        with config_from_dict({'num_particles_rollout_full': 200}):
            # Code that uses the temporary config
            particles = get('num_particles_rollout_full')
        # Config is automatically restored
    """
    # Store original config
    original_config = get_all()
    
    try:
        # Set temporary config
        update(**config_dict)
        yield
    finally:
        # Restore original config
        _config.clear()
        _config.update(original_config) 