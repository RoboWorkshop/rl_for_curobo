#!/usr/bin/env python3
"""
Example 3: Distributed ROS 2 Multi-Arm Goal Coordinator & Monitor.

When running N decentralized `MRStormDistributedArmNode` instances (via
`ros2 launch mr_storm/ros2/launch/distributed_multi_arm.launch.py`), this script
publishes crossing end-effector goals to `/arm_<i>/goal_pose` (`geometry_msgs/PoseStamped`)
and listens to `/mr_storm/plan_broadcast` to monitor live inter-arm plan sharing.
"""

from __future__ import annotations

import argparse
import time
import numpy as np

from mr_storm.core.types import BroadcastPlanPacket


def main():
    parser = argparse.ArgumentParser(description="Send crossing goals to distributed ROS 2 MR-STORM nodes.")
    parser.add_argument("--num_arms", type=int, default=4, help="Number of distributed arms.")
    args = parser.parse_args()

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from geometry_msgs.msg import PoseStamped
    from std_msgs.msg import ByteMultiArray

    rclpy.init()
    node = Node("mr_storm_goal_publisher")

    goal_pubs = {
        i: node.create_publisher(PoseStamped, f"/arm_{i}/goal_pose", 10)
        for i in range(args.num_arms)
    }

    latest_broadcasts = {}

    def on_plan(msg):
        pkt = BroadcastPlanPacket.from_bytes(bytes(msg.data))
        latest_broadcasts[pkt.robot_id] = pkt

    plan_qos = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
    )
    node.create_subscription(ByteMultiArray, "/mr_storm/plan_broadcast", on_plan, plan_qos)

    # Publish antipodal crossing targets in the shared workspace
    radius = 0.35
    for i in range(args.num_arms):
        angle = (i * 2.0 * np.pi / args.num_arms) + np.pi  # opposite side of workspace
        msg = PoseStamped()
        msg.header.frame_id = "world"
        msg.pose.position.x = float(radius * np.cos(angle))
        msg.pose.position.y = float(radius * np.sin(angle))
        msg.pose.position.z = 0.40
        msg.pose.orientation.w = 0.0
        msg.pose.orientation.x = 1.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        goal_pubs[i].publish(msg)
        node.get_logger().info(
            f"Published goal to /arm_{i}/goal_pose -> ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f}, 0.40)"
        )

    print("Monitoring decentralized /mr_storm/plan_broadcast packets (Ctrl+C to exit)...")
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.2)
            if latest_broadcasts:
                status = " | ".join(
                    f"Arm {r_id}: err={pkt.pos_error:.3f}m (H={pkt.horizon}, S={pkt.n_spheres})"
                    for r_id, pkt in sorted(latest_broadcasts.items())
                )
                print(f"\r[Plan-Broadcast Bus] {status}", end="", flush=True)
    except KeyboardInterrupt:
        print("\nExiting monitor.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
