from contextlib import contextmanager
from typing import Dict, Any
from .. import config

@contextmanager
def temporary_config(**kwargs):
    """Context manager for temporary configuration changes.
    
    Usage:
        with temporary_config(num_particles_rollout_full=200, horizon_rollout_full=75):
            # Code that uses the temporary config
            particles = config.get('num_particles_rollout_full')
            horizon = config.get('horizon_rollout_full')
        # Config is automatically restored
    """
    # Store original values
    original_values = {}
    for key in kwargs:
        original_values[key] = config.get(key)
    
    # Set temporary values
    try:
        config.update(**kwargs)
        yield
    finally:
        # Restore original values
        config.update(**original_values)

@contextmanager
def config_from_dict(config_dict: Dict[str, Any]):
    """Context manager to temporarily use a different configuration.
    
    Usage:
        with config_from_dict({'num_particles_rollout_full': 200}):
            # Code that uses the temporary config
            particles = config.get('num_particles_rollout_full')
    """
    # Store original config
    original_config = config.get_all()
    
    try:
        # Set temporary config
        config.update(**config_dict)
        yield
    finally:
        # Restore original config
        config.reset()
        config.update(**original_config) 