#!/usr/bin/env python3
import subprocess
import os
import sys
import argparse
import time
import psutil
import signal

# Define server scripts relative to the root of the repository
SERVER_SCRIPTS = {
    "reader": "projects_root/real_world/servers/reader_server.py",
    "writer": "projects_root/real_world/servers/writer_server.py"
}

SOCKET_FILES = [
    "/tmp/lowstate.sock",
    "/tmp/lowcmd.sock"
]

def get_root_dir():
    """Calculate the root directory (parent of projects_root)."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # projects_root/real_world/servers/start_servers.py -> needs to go up 3 levels
    return os.path.abspath(os.path.join(current_dir, "../../../"))

def find_server_processes():
    """Find running server processes by command line analysis."""
    detected_procs = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['cmdline']:
                cmdline = " ".join(proc.info['cmdline'])
                for name, script_path in SERVER_SCRIPTS.items():
                    if script_path in cmdline and "python" in proc.info['name']:
                        detected_procs.append((name, proc))
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return detected_procs

def start_servers():
    """Start the servers if not already running."""
    root_dir = get_root_dir()
    existing_procs = find_server_processes()
    
    if existing_procs:
        print("Some servers are already running:")
        for name, proc in existing_procs:
            print(f"  - {name} (PID {proc.pid})")
        print("Please stop them first or use 'restart'.")
        return

    print(f"Launching servers from: {root_dir}")
    processes = []
    
    for name, script_path in SERVER_SCRIPTS.items():
        base_command = ["python3", script_path]
        try:
            print(f"Starting {name} server: {' '.join(base_command)}")
            p = subprocess.Popen(base_command, cwd=root_dir)
            processes.append(p)
            print(f"Started {name} server with PID: {p.pid}")
        except Exception as e:
            print(f"Failed to start {name} server: {e}")
    
    return processes

def stop_servers():
    """Stop all running server processes and clean up sockets."""
    print("Stopping servers...")
    procs = find_server_processes()
    
    if not procs:
        print("No servers found running.")
    else:
        for name, proc in procs:
            try:
                print(f"Killing {name} (PID {proc.pid})...")
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except psutil.TimeoutExpired:
                    print(f"Force killing {name}...")
                    proc.kill()
            except psutil.NoSuchProcess:
                print(f"{name} already dead.")
            except Exception as e:
                print(f"Error killing {name}: {e}")

    # Clean up socket files
    for sock_file in SOCKET_FILES:
        if os.path.exists(sock_file):
            try:
                os.remove(sock_file)
                print(f"Removed socket file: {sock_file}")
            except Exception as e:
                print(f"Error removing {sock_file}: {e}")

def check_status():
    """Check and print the status of servers."""
    print("Checking server status...")
    procs = find_server_processes()
    
    if procs:
        print("Running servers:")
        for name, proc in procs:
            status = proc.status()
            print(f"  - {name}: PID {proc.pid} ({status})")
    else:
        print("No servers running.")
    
    print("\nSocket files:")
    for sock_file in SOCKET_FILES:
        if os.path.exists(sock_file):
            print(f"  - {sock_file}: Exists")
        else:
            print(f"  - {sock_file}: Missing")

def main():
    parser = argparse.ArgumentParser(description="Manage Robot Reader/Writer Servers")
    parser.add_argument("action",  nargs="?", choices=["start", "stop", "restart", "status"], default="start",
                        help="Action to perform (default: start)")
    parser.add_argument("--background", "-b", action="store_true", help="Run in background (do not block)")
    
    args = parser.parse_args()
    
    if args.action == "start":
        start_servers()
        if not args.background:
            print("\nServers are running. Press Ctrl+C to stop all servers.")
            try:
                # Keep the script running to catch Ctrl+C
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nReceived interrupt. Stopping servers...")
                stop_servers()
    elif args.action == "stop":
        stop_servers()
    elif args.action == "restart":
        stop_servers()
        time.sleep(1)
        start_servers()
        # For restart, we also block unless background is specified, assuming standard usage pattern
        if not args.background:
             print("\nServers restarted. Press Ctrl+C to stop all servers.")
             try:
                while True:
                    time.sleep(1)
             except KeyboardInterrupt:
                print("\nReceived interrupt. Stopping servers...")
                stop_servers()
    elif args.action == "status":
        check_status()

if __name__ == "__main__":
    main()
