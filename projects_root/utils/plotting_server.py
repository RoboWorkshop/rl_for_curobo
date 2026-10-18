#!/usr/bin/env python3
"""
Plotting server that runs CentralizedLivePlotter in a separate subprocess.
This completely isolates the matplotlib GUI from the main simulation process.
"""

import sys
import time
import pickle
import signal
import multiprocessing as mp
from multiprocessing import Queue, Process, Event
import traceback
import os

# Add the curobo path dynamically relative to repository root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'curobo', 'src')))

def plotting_server_process(data_queue, shutdown_event, max_agents: int = 4):
    """
    Main function that runs in the plotting subprocess.
    Receives data from the main process and updates plots.
    """
    
    print(f"Plotting server starting with max_agents={max_agents}")
    
    try:
        # Import matplotlib and set up the backend
        import matplotlib
        
        # Try different backends in order of preference
        backends_to_try = ['Qt5Agg', 'TkAgg', 'GTK3Agg']
        backend_set = False
        
        for backend in backends_to_try:
            try:
                matplotlib.use(backend)
                import matplotlib.pyplot as plt
                
                # Test if backend actually works
                fig = plt.figure()
                plt.close(fig)
                
                backend_set = True
                break
            except Exception:
                continue
        
        if not backend_set:
            import matplotlib.pyplot as plt
        
        # Force interactive mode
        plt.ion()
        
        # Import and create the centralized plotter
        from curobo.rollout.arm_reacher import CentralizedLivePlotter
        plotter = CentralizedLivePlotter()
        plotter.enable_plotting(max_agents=max_agents)
        
        print(f"✓ Plotting server initialized successfully")
        
        # Set up graceful shutdown
        def signal_handler(signum, frame):
            shutdown_event.set()
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        # Main loop: process data and update plots
        last_update = time.time()
        update_frequency = 0.1  # Update every 100ms
        
        while not shutdown_event.is_set():
            try:
                # Process all available data from queue
                data_processed = False
                while not data_queue.empty() and not shutdown_event.is_set():
                    try:
                        # Get data with timeout to avoid blocking
                        data = data_queue.get(timeout=0.01)
                        
                        if data is None:  # Shutdown signal
                            shutdown_event.set()
                            break
                            
                        # Add data to plotter
                        agent_id = data.get('agent_id', 0)
                        cost_dict = data.get('costs', {})
                        plotter.add_data(agent_id, cost_dict)
                        data_processed = True
                        
                    except Exception as e:
                        # Skip problematic data
                        continue
                
                # Update plots if we got new data and enough time has passed
                current_time = time.time()
                if data_processed and (current_time - last_update) >= update_frequency:
                    plotter.update_plots()
                    last_update = current_time
                
                # Small sleep to prevent busy waiting
                time.sleep(0.01)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error in plotting server: {e}")
                continue
    
    except Exception as e:
        print(f"Fatal error in plotting server: {e}")
        traceback.print_exc()
    
    finally:
        print("Plotting server shutting down...")
        try:
            # Clean shutdown
            if 'plotter' in locals():
                plotter.shutdown()
            plt.close('all')
        except Exception as e:
            print(f"Error during shutdown: {e}")
        print("✓ Plotting server shutdown complete")


class PlottingServerManager:
    """
    Manager for the plotting server subprocess.
    Handles starting, stopping, and communicating with the plotting server.
    """
    
    def __init__(self, max_agents: int = 4):
        self.max_agents = max_agents
        self.process = None
        self.data_queue = None
        self.shutdown_event = None
        self.is_running = False
    
    def start(self):
        """Start the plotting server subprocess."""
        if self.is_running:
            print("⚠ Plotting server already running")
            return
        
        try:
            # Create communication channels
            self.data_queue = Queue(maxsize=1000)  # Large queue to prevent blocking
            self.shutdown_event = Event()
            
            # Start the plotting subprocess
            self.process = Process(
                target=plotting_server_process,
                args=(self.data_queue, self.shutdown_event, self.max_agents),
                daemon=False  # Don't make it daemon so it can show GUI
            )
            
            self.process.start()
            self.is_running = True
            
            print(f"✓ Plotting server started (PID: {self.process.pid})")
            
            # Give it a moment to initialize
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Failed to start plotting server: {e}")
            self.is_running = False
    
    def send_data(self, agent_id: int, cost_dict: dict):
        """Send cost data to the plotting server."""
        if not self.is_running or self.data_queue is None:
            return
        
        try:
            # Convert tensors to plain values
            cost_data = {}
            
            for cost_name, cost_value in cost_dict.items():
                try:
                    # Handle torch tensors
                    if hasattr(cost_value, 'cpu'):
                        import torch
                        cost_data[cost_name] = torch.mean(cost_value).cpu().numpy().item()
                    else:
                        cost_data[cost_name] = float(cost_value)
                except Exception:
                    continue  # Skip problematic values
            
            # Send data (non-blocking)
            data = {
                'agent_id': agent_id,
                'costs': cost_data,
                'timestamp': time.time()
            }
            
            try:
                self.data_queue.put_nowait(data)
            except Exception:
                # Queue full, skip this data point
                pass
                
        except Exception:
            # Don't let plotting errors break the simulation
            pass
    
    def stop(self):
        """Stop the plotting server subprocess."""
        if not self.is_running:
            return
        
        print("Stopping plotting server...")
        
        try:
            # Signal shutdown
            if self.shutdown_event:
                self.shutdown_event.set()
            
            # Send shutdown signal via queue
            if self.data_queue:
                try:
                    self.data_queue.put_nowait(None)
                except Exception:
                    pass
            
            # Wait for process to finish
            if self.process and self.process.is_alive():
                # print(f"debug: plotting server is alive, joining...")
                # self.process.join(timeout=5.0)
                self.process.join(timeout=0.01)
                if self.process.is_alive():
                    print("⚠ Plotting server didn't shut down gracefully, terminating...")
                    self.process.terminate()
                    self.process.join(timeout=2.0)
                    
                    if self.process.is_alive():
                        print("⚠ Force killing plotting server...")
                        self.process.kill()
                        self.process.join()




                             

            self.is_running = False
            print("✓ Plotting server stopped")
            
        except Exception as e:
            print(f"Error stopping plotting server: {e}")
            self.is_running = False


# Global instance for easy access
_global_plotting_server = None

def get_plotting_server() -> PlottingServerManager:
    """Get the global plotting server instance."""
    global _global_plotting_server
    if _global_plotting_server is None:
        _global_plotting_server = PlottingServerManager()
    return _global_plotting_server

def start_plotting_server(max_agents: int = 4):
    """Start the global plotting server."""
    server = get_plotting_server()
    server.max_agents = max_agents
    server.start()

def stop_plotting_server():
    """Stop the global plotting server."""
    server = get_plotting_server()
    server.stop()

def send_plot_data(agent_id: int, cost_dict: dict):
    """Send data to the plotting server."""
    server = get_plotting_server()
    server.send_data(agent_id, cost_dict)


if __name__ == "__main__":
    # Test the plotting server
    print("Testing plotting server...")
    
    try:
        # Start server
        start_plotting_server(max_agents=2)
        
        # Send some test data
        import torch
        test_costs = {
            'total': torch.tensor([10.0, 8.0, 6.0]),
            'goal': torch.tensor([5.0, 4.0, 3.0]),
            'collision': torch.tensor([2.0, 1.5, 1.0])
        }
        
        print("Sending test data...")
        for i in range(100):
            send_plot_data(0, test_costs)
            send_plot_data(1, test_costs)
            time.sleep(0.1)
        
        print("Test complete. Press Ctrl+C to stop.")
        
        # Keep running until interrupted
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
    
    finally:
        stop_plotting_server() 