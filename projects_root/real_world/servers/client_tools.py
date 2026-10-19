
"""
client tools for real world robot control
"""

import socket
import json


def get_joint_states(socket_path='/tmp/lowstate.sock'):
    """
    Get current joint positions from the server.
    
    Returns:
        list: Joint positions, or None if error
    """
    try:
        # Create socket and connect to server
        client_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client_socket.connect(socket_path)
        
        # Send request (can be any message)
        client_socket.sendall(b'get_state')
        
        # Receive response
        data = client_socket.recv(4096)
        response = json.loads(data.decode('utf-8'))
        
        # Close connection
        client_socket.close()
        
        full_joint_positions = response['joint_positions'] # 35 length
        full_joint_velocities = response['joint_velocities'] # 35 length

        left_arm_positions, right_arm_positions = filter_joint_states(full_joint_positions) # 7 dof each
        left_arm_velocities, right_arm_velocities = filter_joint_states(full_joint_velocities) # 7 dof each
        assert len(left_arm_positions) == 7, f"left arm has {len(left_arm_positions)} dof: {left_arm_positions}"
        assert len(right_arm_positions) == 7, f"right arm has {len(right_arm_positions)} dof: {right_arm_positions}"
        assert len(left_arm_velocities) == 7, f"left arm has {len(left_arm_velocities)} dof: {left_arm_velocities}"
        assert len(right_arm_velocities) == 7, f"right arm has {len(right_arm_velocities)} dof: {right_arm_velocities}"
        return (left_arm_positions, left_arm_velocities), (right_arm_positions, right_arm_velocities)
        
    except Exception as e:
        print(f"Error getting joint states: {e}")
        return None

def get_joint_velocities(socket_path='/tmp/lowstate.sock'):
    """
    Get current joint velocities from the server.
    
    Returns:
        list: Joint velocities, or None if error
    """
    try:
        # Create socket and connect to server
        client_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client_socket.connect(socket_path)
        
        # Send request (can be any message)
        client_socket.sendall(b'get_state')
        
        # Receive response
        data = client_socket.recv(4096)
        response = json.loads(data.decode('utf-8'))
        
        # Close connection
        client_socket.close()
        
        full_js = response['joint_velocities'] # 35 length
        left_js, right_js = filter_joint_states(full_js) # 7 dof each
        assert len(left_js) == 7, f"left arm has {len(left_js)} dof: {left_js}"
        assert len(right_js) == 7, f"right arm has {len(right_js)} dof: {right_js}"
        return left_js, right_js
        
    except Exception as e:
        print(f"Error getting joint velocities: {e}")
        return None

def send_joint_commands(joint_positions:list[float], arm_idx:int, socket_path='/tmp/lowcmd.sock'):
    """
    Send desired joint positions to the robot via the ROS2 server process.
    
    Args:
        joint_positions: List of 7 desired joint positions
        arm_idx: 0 for left, 1 for right
        socket_path: Path to the Unix socket
    Returns:
        dict: Response with 'success' (bool) and 'message' (str)
    """
    try:
        # Create socket and connect to server
        client_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client_socket.connect(socket_path)
        
        # Prepare request
        request = json.dumps({
            'joint_positions': joint_positions,
            'arm_idx': arm_idx
        })
        
        # Send request
        client_socket.sendall(request.encode('utf-8'))
        
        # Receive response
        data = client_socket.recv(4096)
        response = json.loads(data.decode('utf-8'))
        
        # Close connection
        client_socket.close()
        
        return response
    
    except Exception as e:
        print(f"Error sending joint commands: {e}")
        return {
            'success': False,
            'message': str(e)
        }

def filter_joint_states(full_js) -> tuple[list[float], list[float]]:
    """
    Filter joint states for left and right arms from full joint states.
    
    Args:
        full_js: List of 35 joint positions (full robot (29) + padding at the end (6)). ONLY FOR 29 DOF G1, SEE TABLES https://support.unitree.com/home/en/G1_developer
    
    Returns:
        tuple: (left_arm_js, right_arm_js)
        each arm has 7 dof following this order: (shoulder pitch, shoulder roll, shoulder yaw, elbow, wrist roll, wrist pitch, wrist yaw)
    """
    left_arm_js = full_js[15:22] # 7 dof 
    right_arm_js = full_js[22:29] # 7 dof  
    return left_arm_js, right_arm_js

if __name__ == '__main__':
    # Example usage - call this in your loop
    import time
    print("Requesting joint states from server...")
    i = 0
    while True:
        
        (left_arm_positions, left_arm_velocities), (right_arm_positions, right_arm_velocities) = get_joint_states()
        # next_js = mpc.plan()
        # left_arm_js, right_arm_js = filter_joint_states(full_js)
        
        # send_joint_commands(left_arm_positions, 0)
        # print(f'left sent: {left_arm_positions}')
        send_joint_commands(left_arm_positions, 0)
        send_joint_commands(right_arm_positions, 1)
        # print(f'right sent: {right_arm_positions}')
        print(f"\n # n = {i+1} requests completed")
        print(f"client: left_js: {left_arm_positions}")
        # print(f"client: left_velocities: {left_arm_velocities}")
        print(f"client: right_js: {right_arm_positions}")
        # print(f"client: right_velocities: {right_arm_velocities}")
        i += 1
        # time.sleep(0.5)
