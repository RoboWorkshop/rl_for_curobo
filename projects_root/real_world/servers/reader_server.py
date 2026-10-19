#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from unitree_hg.msg import LowState
import socket
import os
import json
import threading

        
class LowStateSubscriber(Node):
    """Subscribe to reading ros topic of joint states"""
    def __init__(self):
        super().__init__('lowstate_subscriber')
        self.subscription = self.create_subscription(
            LowState,
            '/lowstate',
            self.lowstate_callback,
            10
        )
        self.subscription  # prevent unused variable warning
        self.latest_joint_positions = None
        self.get_logger().info('LowState subscriber initialized')
        
    def lowstate_callback(self, msg):
        # Store only joint positions
        self.latest_joint_positions = [motor.q for motor in msg.motor_state]
        self.latest_joint_velocities = [motor.dq for motor in msg.motor_state]

    def get_joint_positions(self):
        """Return the latest joint positions"""
        return self.latest_joint_positions
    
    def get_joint_velocities(self):
        """Return the latest joint velocities"""
        return self.latest_joint_velocities


class LowStateServer:
    """Receives get_joint_state request from client"""
    def __init__(self, socket_path='/tmp/lowstate.sock'):
        self.socket_path = socket_path
        self.response_cntr = 0
        
        # Remove socket file if it exists
        if os.path.exists(self.socket_path):
            os.remove(self.socket_path)
        
        # Initialize ROS2
        rclpy.init() # listen to real joint states from robot
        self.subscriber = LowStateSubscriber()
        
        # Create Unix domain socket server
        self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_socket.bind(self.socket_path)
        self.server_socket.listen(5)
        
        print(f"LowState server listening on {self.socket_path}")
        
        # Start ROS2 spin in separate thread
        self.spin_thread = threading.Thread(target=self._spin_ros, daemon=True)
        self.spin_thread.start()
        
        # Start accepting connections
        self.run()
    
    def _spin_ros(self):
        """Spin ROS2 in background thread"""
        while rclpy.ok():
            # print("Spinning ROS2...")
            rclpy.spin_once(self.subscriber, timeout_sec=0.001)
    
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
            # Receive request (we don't really need to parse it, just respond)
            data = client_socket.recv(1024)
            
            if data:
                # Get latest joint positions
                joint_positions = self.subscriber.get_joint_positions() # full body (35)
                joint_velocities = self.subscriber.get_joint_velocities() # full body (35)

                # Send response as JSON
                response = json.dumps({
                    'joint_positions': joint_positions,
                    'joint_velocities': joint_velocities
                })
                client_socket.sendall(response.encode('utf-8'))
                
                self.response_cntr += 1
                print(f"debug: sent response")
                print(f"debug: joint_positions: {joint_positions}")
                print(f"debug: joint_velocities: {joint_velocities}")
                print(f"debug: response: {self.response_cntr}")
                
        except Exception as e:
            print(f"Error handling client: {e}")
        finally:
            client_socket.close()
    
    def cleanup(self):
        """Cleanup resources"""
        self.server_socket.close()
        if os.path.exists(self.socket_path):
            os.remove(self.socket_path)
        self.subscriber.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    server = LowStateServer()
