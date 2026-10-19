# G1 Robot Configuration & Debugging Guidelines

This document provides best practices and guidelines for setting up new G1 robot configurations in Isaac Sim/Curobo, based on lessons learned from debugging the "wiggling" arm instability.

## 1. URDF Dynamics (Crucial for Stability)
When creating a new URDF or modifying an existing one, especially for lightweight robots like the G1:
- **Always Include Dynamics Tags:** Joints must have `<dynamics damping="..." friction="..."/>` defined.
- **Why?** The G1 has very lightweight links (e.g., wrist links ~0.08kg). Without explicit damping, the physics engine (PhysX) treats them as frictionless, leading to undamped high-frequency oscillations ("wiggling") under motor control.
- **Recommended Values for G1:**
  - *Shoulder/Elbow:* `damping="1.0"`, `friction="0.5"`
  - *Wrist/Hand:* `damping="0.5"`, `friction="0.1"`
- **Symptom of Missing Dynamics:** The robot shakes or vibrates uncontrollably even when holding a steady pose, regardless of controller gains.

## 2. Self-Collision Configuration
Complex robotic assemblies (like 7-DOF arms/wrists) often have overlapping geometry.
- **Identify Overlaps:** Visualize collision spheres to see which links naturally intersect.
- **Use Ignore Rules:** Explicitly ignore collisions between adjacent links that are mechanically connected but whose collision volumes overlap. Do this in the robot's YAML config (e.g., `g1_feb26/left_arm.yml`), **NOT** by setting global collision weights to zero.
  ```yaml
  self_collision_ignore:
    "left_wrist_roll_link": ["left_wrist_pitch_link", "left_wrist_yaw_link"]
  ```
- **Don't Disable Physics:** Setting `self_collision_cfg.weight: 0` in the MPC config hides the problem but causes the planner to generate invalid, self-colliding trajectories that the physics engine then fights against, causing instability.

## 3. Simulation & Multiprocessing
When running complex simulations with Curobo and Isaac Sim:
- **CUDA Context Management:** Python's `multiprocessing` default (fork) can cause CUDA initialization errors ("operation failed due to a previous error").
- **Fix:** Always set the start method to `spawn` *before* importing heavy libraries or starting the simulation app.
  ```python
  import multiprocessing
  multiprocessing.set_start_method('spawn', force=True)
  ```
- **Avoid Double Launch:** Place `SimulationApp` initialization inside `if __name__ == "__main__":` blocks to prevent spawned processes from re-launching the simulator.

## 4. Resolving Collision Penetration
If robot parts are passing through each other (e.g. arm through torso):
- **Check URDF Collision Geometry:** Ensure all links have `<collision>` tags defined. Visual meshes alone are not enough for physics!
  - *Example:* `waist_yaw_link` was missing collision tags, making it a "ghost" to the physics engine.
- **Update Collision Config (`robot.yml`):**
  - Add all relevant links to `collision_link_names`.
  - Ensure correct spheres are defined in the sphere config file.
- **Verify Ignore Rules:** Be careful with `self_collision_ignore`.
  - Collision ignores are often symmetric. If `Link A` ignores `Link B`, the system might allow collision even if `Link B` doesn't list `Link A`.
  - *Fix:* Remove both links from each other's ignore lists to enforce collision checking.

## Debugging Checklist
If the robot is behaving erratically:
1.  **Check URDF Dynamics:** Are `damping` and `friction` non-zero?
2.  **Check Collision Ignores:** Are adjacent links causing constant collisions?
3.  **Check Mass Properties:** Are masses realistic? (Too low mass = instability).
4.  **Verify Controller:** Are gains reachable? (Don't hack gains to fix physics issues; fix the physics model first).
5.  **Check Collision Geometry:** Do all links have `<collision>` tags in URDF?
