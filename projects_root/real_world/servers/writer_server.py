#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import socket
import os
import json
import threading
import time

# Joint names must match the order in joint_positions [shoulder pitch, roll, yaw, elbow, wrist roll, pitch, yaw]
# These names are based on the user's config
JOINT_NAMES_LEFT = [
    'left_shoulder_pitch_joint', 
    'left_shoulder_roll_joint', 
    'left_shoulder_yaw_joint', 
    'left_elbow_joint', 
    'left_wrist_roll_joint', 
    'left_wrist_pitch_joint', 
    'left_wrist_yaw_joint'
]

JOINT_NAMES_RIGHT = [
    'right_shoulder_pitch_joint', 
    'right_shoulder_roll_joint', 
    'right_shoulder_yaw_joint', 
    'right_elbow_joint', 
    'right_wrist_roll_joint', 
    'right_wrist_pitch_joint', 
    'right_wrist_yaw_joint'
]

class LowCmdPublisher(Node):
    """Publish actions (target joint states) to ros topic"""
    def __init__(self):
        super().__init__('lowcmd_publisher')
        
        # JointTrajectory publisher
        self.traj_publisher = self.create_publisher(
            JointTrajectory,
            '/arm_plan',
            200 # Hz
        )
        
        self.get_logger().info('LowCmd publisher initialized for /arm_plan')
    
    def send_joint_commands(self, joint_positions, arm_idx):
        """
        Publish joint position commands to the robot.
        
        Args:
            joint_positions: List of 7 desired (target) joint positions
            arm_idx: 0 for left arm, 1 for right arm
        """
        
        # Select joint names based on arm_idx
        if arm_idx == 0:
            joint_names = JOINT_NAMES_LEFT
        elif arm_idx == 1:
            joint_names = JOINT_NAMES_RIGHT
        else:
            self.get_logger().error(f'Invalid arm_idx: {arm_idx}')
            return False
            
        # Publish using JointTrajectory format to /arm_plan
        try:
            traj_msg = JointTrajectory()
            traj_msg.header.stamp = self.get_clock().now().to_msg()
            traj_msg.joint_names = joint_names
            
            point = JointTrajectoryPoint()
            point.positions = [float(p) for p in joint_positions]
            point.velocities = [0.0] * len(joint_positions)
            point.accelerations = [0.0] * len(joint_positions)
            point.effort = [0.0] * len(joint_positions)
            point.time_from_start.sec = 0
            point.time_from_start.nanosec = 0 
            
            traj_msg.points.append(point)
            
            self.traj_publisher.publish(traj_msg)
            
        except Exception as e:
            self.get_logger().error(f'Failed to publish trajectory: {e}')
            return False
            
        return True


class LowCmdServer:
    """Receives action requests from client and publishes them to ros topic for processing"""
    def __init__(self, socket_path='/tmp/lowcmd.sock'):
        self.socket_path = socket_path
        
        # Remove socket file if it exists
        if os.path.exists(self.socket_path):
            os.remove(self.socket_path)
        
        # Initialize ROS2
        rclpy.init()
        self.publisher = LowCmdPublisher()
        
        # Create Unix domain socket server
        self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_socket.bind(self.socket_path)
        self.server_socket.listen(5)
        
        print(f"LowCmd server listening on {self.socket_path}")
        
        # Start ROS2 spin in separate thread
        self.spin_thread = threading.Thread(target=self._spin_ros, daemon=True)
        self.spin_thread.start()
        
        # Start accepting connections
        self.run()
    
    def _spin_ros(self):
        """Spin ROS2 in background thread"""
        while rclpy.ok():
            # rclpy.spin_once(self.publisher, timeout_sec=0.005)
            rclpy.spin_once(self.publisher)
                
    def run(self):
        """Accept connections and handle requests"""
        try:
            while True:
                # Accept new connection
                client_socket, _ = self.server_socket.accept()
                
                # Handle request in separate thread
                threading.Thread(
                    target=self._handle_client,
                    args=(client_socket,),
                    daemon=True
                ).start()
        except KeyboardInterrupt:
            print("\nShutting down server...")
        finally:
            self.cleanup()
    
    def _handle_client(self, client_socket):
        """Handle individual client request"""
        try:
            # Receive command data
            data = client_socket.recv(8192)  # Larger buffer for joint commands
            
            if data:
                # Parse JSON request
                request = json.loads(data.decode('utf-8'))
                joint_positions = request.get('joint_positions') # target joint positions (order = [shoulder pitch, shoulder roll, shoulder yaw, elbow, wrist roll, wrist pitch, wrist yaw])
                arm_idx = request.get('arm_idx') # 0 for left arm, 1 for right arm
                
                # Send command to robot
                success = self.publisher.send_joint_commands(joint_positions, arm_idx)
                
                # Send acknowledgment back to client
                response = json.dumps({
                    'success': success,
                    'message': 'Command sent' if success else 'Command failed'
                })
                client_socket.sendall(response.encode('utf-8'))
        except Exception as e:
            print(f"Error handling client: {e}")
            # Send error response
            try:
                error_response = json.dumps({
                    'success': False,
                    'message': str(e)
                })
                client_socket.sendall(error_response.encode('utf-8'))
            except:
                pass
        finally:
            client_socket.close()
    
    def cleanup(self):
        """Cleanup resources"""
        self.server_socket.close()
        if os.path.exists(self.socket_path):
            os.remove(self.socket_path)
        self.publisher.destroy_node()
        rclpy.shutdown()
        
if __name__ == '__main__':
    server = LowCmdServer()
