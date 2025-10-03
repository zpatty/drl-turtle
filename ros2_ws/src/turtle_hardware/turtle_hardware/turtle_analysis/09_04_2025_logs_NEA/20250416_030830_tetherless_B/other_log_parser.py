import numpy as np
import re
import json
import os
from datetime import datetime
import argparse

class TurtleLogParser:
    def __init__(self):
        self.sensor_data = []
        self.motor_inputs = []
        self.stereo_depths = []
        self.headings = []
        self.fps_data = []
        self.timestamps = []
        self.nav_commands = []
        self.stereo_debug_depths = []
        self.turn_commands = []
        self.turn_command = None
        # Regex patterns for different log types
        self.sensor_pattern = r'\[INFO\] \[(\d+\.\d+)\] \[turtle_sensors_node\]: (.+)'
        self.motor_pattern = r'input: (\[.+\])'
        self.fps_pattern = r'fps: ([\d.]+)'
        self.heading_pattern = r'heading: ([-\d.]+)'
        self.stereo_debug_pattern = r'\[DEBUG\] stereo depth: ([\d.]+)'
        self.turn_pattern = r'\[DEBUG\] (turn \w+)'

        self.depth_value = None
        self.flag_turn = None
        # to regenerate yaw_d
        self.yaw_d = 2.2
        self.yaw_ds = [self.yaw_d]
        # self.depth_pattern = r'^([\d.]+)
        self.depth_pattern = r'^([\d.]+)$'  # Single depth values
        
    def parse_timestamp(self, timestamp_str):
        """Convert ROS timestamp to seconds"""
        return float(timestamp_str)
    
    def parse_sensor_line(self, timestamp, data_str):
        """Parse sensor JSON data"""
        try:
            # Clean up the data string and parse as JSON
            if data_str.strip().startswith('{') and data_str.strip().endswith('}'):
                sensor_json = json.loads(data_str)
                
                entry = {
                    'timestamp': timestamp,
                    'quat': sensor_json.get('Quat', [0, 0, 0, 0]),
                    'acc': sensor_json.get('Acc', [0, 0, 0]),
                    'gyr': sensor_json.get('Gyr', [0, 0, 0]),
                    'depth': sensor_json.get('Depth', [0])[0] if sensor_json.get('Depth') else 0,
                    'altitude': sensor_json.get('Altitude', [0, 0]),
                    'voltage': sensor_json.get('Voltage', [0])[0] if sensor_json.get('Voltage') else 0
                }
                self.sensor_data.append(entry)
                return True
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            # Skip malformed sensor data
            pass
        return False
    
    def parse_motor_line(self, line):
        """Parse motor input commands"""
        try:
            motor_match = re.search(self.motor_pattern, line)
            if motor_match:
                motor_str = motor_match.group(1)
                # Parse the array string
                motor_values = eval(motor_str)  # Safe here since we control the format
                if len(motor_values) == 10:  # Ensure we have 10 motor values
                    self.motor_inputs.append(motor_values)
                    return True
        except (SyntaxError, ValueError):
            pass
        return False
    
    def parse_fps_line(self, line):
        """Parse FPS data"""
        try:
            fps_match = re.search(self.fps_pattern, line)
            if fps_match:
                fps_value = float(fps_match.group(1))
                self.fps_data.append(fps_value)
                return True
        except ValueError:
            pass
        return False
    
    def parse_heading_line(self, line):
        """Parse heading data"""
        try:
            heading_match = re.search(self.heading_pattern, line)
            if heading_match:
                heading_value = float(heading_match.group(1))
                self.headings.append(heading_value)
                self.heading = heading_value
                return True
        except ValueError:
            pass
        return False
    def parse_depth_line(self, line):
        """Parse standalone depth values"""
        try:
            depth_match = re.search(self.depth_pattern, line.strip())
            if depth_match and not line.strip().startswith('fps:'):
                depth_value = float(depth_match.group(1))
                # Filter out obviously invalid depths (like 10000000.0)
                self.stereo_depths.append(depth_value)
                self.depth_value = depth_value
                return True
        except ValueError:
            pass
        return False

    def parse_nav_u_line(self, line):
        """Parse navigation control commands"""
        self.nav_u_pattern = r'no flag u: (\[.+\])'

        try:
            nav_match = re.search(self.nav_u_pattern, line)
            if nav_match:
                nav_str = nav_match.group(1)
                # Parse the array string
                nav_values = eval(nav_str)  # Safe here since we control the format
                if len(nav_values) >= 4:  # Ensure we have at least 4 nav values
                    self.nav_commands.append(nav_values[:4])  # Take first 4 for nav_u
                    # also add yaw_d here
                    if self.depth_value is not None:
                        if self.depth_value < 3.5 and not self.flag_turn:
                            self.flag_turn = True
                            u_yaw = nav_values[3]
                            if u_yaw < 0:
                                yaw_d = np.arctan2(np.sin(self.heading + np.deg2rad(120)), np.cos(self.heading + np.deg2rad(120)))
                            elif u_yaw > 0:
                                yaw_d = np.arctan2(np.sin(self.heading - np.deg2rad(120)), np.cos(self.heading - np.deg2rad(120)))
                            else:
                                print("we don't have u yaw for some reason")
                            self.yaw_ds.append(yaw_d)
                        elif self.depth_value > 3.0 and self.flag_turn:
                            self.flag_turn = False
                            yaw_d = self.heading
                            self.yaw_ds.append(yaw_d)
                    return True
            elif self.turn_command != None:
                u = self.nav_commands[-1].copy()
                print(f"self.turn_command: {self.turn_command}")
                if self.turn_command == "left":
                    u_yaw = -1.0
                    u_roll = -1.0
                    u[3] = u_yaw
                    u[1] = u_roll
                    u[4] = -1.0
                else:
                    u_yaw = 1.0
                    u_roll = 1.0
                    u[3] = u_yaw
                    u[1] = u_roll
                    u[4] = -1.0
                self.nav_commands.append(u)  # Take first 4 for nav_u


        except (SyntaxError, ValueError):
            pass
    
    def parse_stereo_debug_line(self, line):
        """Parse stereo debug depth values"""
        try:
            stereo_match = re.search(self.stereo_debug_pattern, line)
            if stereo_match:
                depth_value = float(stereo_match.group(1))
                self.stereo_debug_depths.append(depth_value)
                self.depth_value = depth_value
                return True
        except ValueError:
            pass
        return False
    
    def parse_turn_command_line(self, line):
        """Parse turn commands"""
        try:
            turn_match = re.search(self.turn_pattern, line)
            if turn_match:
                turn_cmd = turn_match.group(1)
                self.turn_commands.append(turn_cmd)
                self.turn_command = turn_cmd
                return True
        except:
            print("Failed to parse turn command")
            pass
        return False
    
    def parse_log_file(self, filepath):
        """Parse a single log file"""
        print(f"Parsing log file: {filepath}")
        
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                # Try to parse sensor data first (JSON format)
                sensor_match = re.search(self.sensor_pattern, line)
                if sensor_match:
                    timestamp = self.parse_timestamp(sensor_match.group(1))
                    # print(f"timestamp: {timestamp}")
                    data_str = sensor_match.group(2)
                    self.parse_sensor_line(timestamp, data_str)
                    continue
                
                # Try to parse motor data
                if 'input:' in line:
                    self.parse_motor_line(line)
                    continue
                
                # Try to parse FPS data
                if 'fps:' in line:
                    self.parse_fps_line(line)
                    continue
                
                # Try to parse heading data
                if 'heading:' in line:
                    self.parse_heading_line(line)
                    continue
                
                # Try to parse navigation commands
                if 'no flag u:' in line:
                    self.parse_nav_u_line(line)
                    continue
                
                # Try to parse stereo debug depth
                if '[DEBUG] stereo depth:' in line:
                    self.parse_stereo_debug_line(line)
                    continue
                
                # Try to parse turn commands
                if '[DEBUG] turn' in line:
                    self.parse_turn_command_line(line)
                    continue
                
                # Try to parse standalone depth values
                self.parse_depth_line(line)
        
        print(f"Parsed {len(self.sensor_data)} sensor entries")
        print(f"Parsed {len(self.motor_inputs)} motor input entries") 
        print(f"Parsed {len(self.fps_data)} FPS entries")
        print(f"Parsed {len(self.stereo_depths)} depth entries")
        print(f"Parsed {len(self.headings)} heading entries")
        print(f"Parsed {len(self.nav_commands)} navigation command entries")
        print(f"Parsed {len(self.stereo_debug_depths)} stereo debug depth entries")
        print(f"Parsed {len(self.turn_commands)} turn command entries")
    
    def synchronize_data(self):
        """Synchronize data streams to common time base"""
        # For navigation logs without explicit timestamps, create synthetic time base
        max_length = max(
            len(self.sensor_data),
            len(self.headings),
            len(self.nav_commands),
            len(self.stereo_debug_depths),
            len(self.motor_inputs)
        )
        
        if max_length == 0:
            print("Warning: No data found for synchronization")
            return
        
        if self.sensor_data:
            # Use sensor data timestamps if available
            sensor_timestamps = np.array([entry['timestamp'] for entry in self.sensor_data])
            start_time = sensor_timestamps[0]
            self.timestamps = sensor_timestamps - start_time
            self.primary_length = len(self.sensor_data)
        else:
            # Create synthetic timestamps based on largest dataset
            self.primary_length = max_length
            # Assume ~10Hz sampling rate for navigation data
            self.timestamps = np.arange(max_length) * 0.1
            start_time = 0
        
        print(f"Data synchronized. Primary length: {self.primary_length}, Duration: {self.timestamps[-1] if len(self.timestamps) > 0 else 0:.2f} seconds")
    
    def create_npz_data(self):
        """Convert parsed data to NPZ format matching original structure"""
        if self.primary_length == 0:
            print("Error: No data to convert")
            return None
        
        n_samples = self.primary_length
        
        # Handle sensor data if available
        if self.sensor_data:
            quat_data = np.array([entry['quat'] for entry in self.sensor_data])
            acc_data = np.array([entry['acc'] for entry in self.sensor_data]) 
            gyr_data = np.array([entry['gyr'] for entry in self.sensor_data])
            depth_data = np.array([entry['depth'] for entry in self.sensor_data])
            altitude_data = np.array([entry['altitude'] for entry in self.sensor_data])
            voltage_data = np.array([entry['voltage'] for entry in self.sensor_data])
            
            # Pad or truncate to match primary length
            if len(quat_data) != n_samples:
                quat_padded = np.zeros((n_samples, 4))
                acc_padded = np.zeros((n_samples, 3))
                gyr_padded = np.zeros((n_samples, 3))
                depth_padded = np.zeros(n_samples)
                altitude_padded = np.zeros((n_samples, 2))
                voltage_padded = np.zeros(n_samples)
                
                min_len = min(len(quat_data), n_samples)
                quat_padded[:min_len] = quat_data[:min_len]
                acc_padded[:min_len] = acc_data[:min_len]
                gyr_padded[:min_len] = gyr_data[:min_len]
                depth_padded[:min_len] = depth_data[:min_len]
                altitude_padded[:min_len] = altitude_data[:min_len]
                voltage_padded[:min_len] = voltage_data[:min_len]
                
                quat_data, acc_data, gyr_data = quat_padded, acc_padded, gyr_padded
                depth_data, altitude_data, voltage_data = depth_padded, altitude_padded, voltage_padded
        else:
            # Create zeros if no sensor data
            quat_data = np.zeros((n_samples, 4))
            acc_data = np.zeros((n_samples, 3))
            gyr_data = np.zeros((n_samples, 3))
            depth_data = np.zeros(n_samples)
            altitude_data = np.zeros((n_samples, 2))
            voltage_data = np.zeros(n_samples)
        
        # Handle navigation data
        if self.headings:
            heading_array = np.array(self.headings)
            yaw_d_data = np.zeros(n_samples)
            min_len = min(len(heading_array), n_samples)
            yaw_d_data[:min_len] = heading_array[:min_len]
        else:
            yaw_d_data = np.zeros(n_samples)
        
        if self.nav_commands:
            nav_array = np.array(self.nav_commands)
            nav_u_data = np.zeros((n_samples, 4))
            min_len = min(len(nav_array), n_samples)
            nav_u_data[:min_len] = nav_array[:min_len]
        else:
            nav_u_data = np.zeros((n_samples, 4))
        
        # Handle stereo depth data
        stereo_depth_data = np.zeros(n_samples)
        if self.stereo_debug_depths:
            stereo_array = np.array(self.stereo_debug_depths)
            min_len = min(len(stereo_array), n_samples)
            stereo_depth_data[:min_len] = stereo_array[:min_len]
        elif self.stereo_depths:  # Fallback to regular stereo depths
            stereo_array = np.array(self.stereo_depths)
            min_len = min(len(stereo_array), n_samples)
            stereo_depth_data[:min_len] = stereo_array[:min_len]
        
        # Handle motor data
        if self.motor_inputs:
            motor_array = np.array(self.motor_inputs)
            u_data = np.zeros((n_samples, 10))
            input_data = np.zeros((n_samples, 10))
            min_len = min(len(motor_array), n_samples)
            u_data[:min_len] = motor_array[:min_len]
            input_data[:min_len] = motor_array[:min_len]
            # Fill remaining with last valid values if available
            if len(motor_array) > 0 and len(motor_array) < n_samples:
                u_data[len(motor_array):] = motor_array[-1]
                input_data[len(motor_array):] = motor_array[-1]
        else:
            u_data = np.zeros((n_samples, 10))
            input_data = np.zeros((n_samples, 10))
        
        # Create synthetic data for missing fields
        q_data = np.zeros((n_samples, 10))  # Joint positions - would need actual data
        dq_data = np.zeros((n_samples, 10))  # Joint velocities - would need actual data  
        qd_data = np.zeros((n_samples, 10))  # Desired joint positions - would need actual data
        dqd_data = np.zeros((n_samples, 10))  # Desired joint velocities - would need actual data
        depth_d_data = np.zeros(n_samples)  # Desired depth - would need actual data
        alt_data = altitude_data[:, 0] if altitude_data.shape[1] > 0 else np.zeros(n_samples)
        stereo_point_data = np.zeros((2, n_samples))  # Stereo points - would need actual data
        print(f"turn commands! {self.turn_commands}")
        # Create NPZ data dictionary
        npz_data = {
            'q': q_data,                    # Joint positions (NEEDS REAL DATA)
            'dq': dq_data,                  # Joint velocities (NEEDS REAL DATA)
            't': self.timestamps,           # Time vector ✓
            'input': input_data,            # Motor inputs ✓
            'u': u_data,                    # Control inputs ✓
            'nav_u': nav_u_data,           # Navigation control ✓
            'qd': qd_data,                 # Desired positions (NEEDS REAL DATA)  
            'dqd': dqd_data,               # Desired velocities (NEEDS REAL DATA)
            'depth': depth_data,           # Depth sensor ✓
            'depth_d': depth_d_data,       # Desired depth (NEEDS REAL DATA)
            'quat': quat_data,             # Quaternion orientation ✓
            'alt': alt_data,               # Altitude ✓
            'yaw_d': yaw_d_data,          # Desired yaw ✓ (from headings)
            'stereo_depth': stereo_depth_data,  # Stereo depth ✓
            'stereo_point': stereo_point_data,  # Stereo points (NEEDS REAL DATA)
            # Additional logged data
            'acc': acc_data,               # Accelerometer ✓
            'gyr': gyr_data,              # Gyroscope ✓  
            'voltage': voltage_data,       # Battery voltage ✓
            'fps': np.array(self.fps_data) if self.fps_data else np.array([]),  # Camera FPS ✓
            'headings': np.array(self.headings) if self.headings else np.array([]),  # Raw headings ✓
            'turn_commands': self.turn_commands,  # Turn commands ✓
        }
        
        return npz_data
    
    def save_npz(self, output_path, data=None):
        """Save data to NPZ file"""
        if data is None:
            data = self.create_npz_data()
        
        if data is None:
            print("Error: No data to save")
            return False
        
        np.savez_compressed(output_path, **data)
        print(f"Data saved to: {output_path}")
        
        # Print summary
        print(f"\nData Summary:")
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                print(f"{key}: shape {value.shape}")
        
        return True

def main():
    parser = argparse.ArgumentParser(description='Convert turtle robot log files to NPZ format')
    parser.add_argument('log_files', nargs='+', help='Path to log files')
    parser.add_argument('-o', '--output', help='Output NPZ file path')
    
    args = parser.parse_args()
    
    # Create parser instance
    log_parser = TurtleLogParser()
    
    # Parse all provided log files
    for log_file in args.log_files:
        if os.path.exists(log_file):
            log_parser.parse_log_file(log_file)
        else:
            print(f"Warning: Log file not found: {log_file}")
    
    if not log_parser.sensor_data and not log_parser.motor_inputs:
        print("Error: No valid data found in log files")
        return
    
    # Synchronize data streams
    log_parser.synchronize_data()
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        # Generate output path based on first log file
        base_name = os.path.splitext(os.path.basename(args.log_files[0]))[0]
        output_path = f"{base_name}_converted.npz"
    
    # Save to NPZ
    success = log_parser.save_npz(output_path)
    
    if success:
        print(f"\n✓ Successfully converted logs to NPZ format!")
        print(f"Use this file with your existing visualization code.")
        print(f"\nNote: Some fields are filled with zeros and marked as 'NEEDS REAL DATA'")
        print(f"You may need to modify the parser to extract these from your specific logs.")

if __name__ == "__main__":
    # Example usage if run directly
    if len(os.sys.argv) == 1:
        print("Usage: python log_parser.py <log_file1> [log_file2] ... [-o output.npz]")
        print("\nExample:")
        print("python log_parser.py robot_log1.log robot_log2.log -o trial_data.npz")
    else:
        main()























        
  # Single depth values
        
    #     # New patterns for navigation log format
    #     self.heading_pattern = r'heading: ([-\d.]+)'
    #     self.nav_u_pattern = r'no flag u: (\[.+\])'
    #     self.stereo_debug_pattern = r'\[DEBUG\] stereo depth: ([\d.]+)'
    #     self.turn_pattern = r'\[DEBUG\] (turn \w+)'
        
    #     # Additional data storage for navigation logs
    #     self.headings = []
    #     self.nav_commands = []
    #     self.stereo_debug_depths = []
    #     self.turn_commands = []
        
    # def parse_timestamp(self, timestamp_str):
    #     """Convert ROS timestamp to seconds"""
    #     return float(timestamp_str)
    
    # def parse_sensor_line(self, timestamp, data_str):
    #     """Parse sensor JSON data"""
    #     try:
    #         # Clean up the data string and parse as JSON
    #         if data_str.strip().startswith('{') and data_str.strip().endswith('}'):
    #             sensor_json = json.loads(data_str)
                
    #             entry = {
    #                 'timestamp': timestamp,
    #                 'quat': sensor_json.get('Quat', [0, 0, 0, 0]),
    #                 'acc': sensor_json.get('Acc', [0, 0, 0]),
    #                 'gyr': sensor_json.get('Gyr', [0, 0, 0]),
    #                 'depth': sensor_json.get('Depth', [0])[0] if sensor_json.get('Depth') else 0,
    #                 'altitude': sensor_json.get('Altitude', [0, 0]),
    #                 'voltage': sensor_json.get('Voltage', [0])[0] if sensor_json.get('Voltage') else 0
    #             }
    #             self.sensor_data.append(entry)
    #             return True
    #     except (json.JSONDecodeError, KeyError, IndexError) as e:
    #         # Skip malformed sensor data
    #         pass
    #     return False
    
    # def parse_motor_line(self, line):
    #     """Parse motor input commands"""
    #     try:
    #         motor_match = re.search(self.motor_pattern, line)
    #         if motor_match:
    #             motor_str = motor_match.group(1)
    #             # Parse the array string
    #             motor_values = eval(motor_str)  # Safe here since we control the format
    #             if len(motor_values) == 10:  # Ensure we have 10 motor values
    #                 self.motor_inputs.append(motor_values)
    #                 return True
    #     except (SyntaxError, ValueError):
    #         pass
    #     return False
    
    # def parse_fps_line(self, line):
    #     """Parse FPS data"""
    #     try:
    #         fps_match = re.search(self.fps_pattern, line)
    #         if fps_match:
    #             fps_value = float(fps_match.group(1))
    #             self.fps_data.append(fps_value)
    #             return True
    #     except ValueError:
    #         pass
    #     return False
    
    # def parse_depth_line(self, line):
    #     """Parse standalone depth values"""
    #     try:
    #         depth_match = re.search(self.depth_pattern, line.strip())
    #         if depth_match and not line.strip().startswith('fps:'):
    #             depth_value = float(depth_match.group(1))
    #             # Filter out obviously invalid depths (like 10000000.0)
    #             if 0 < depth_value < 1000:  # Reasonable depth range
    #                 self.stereo_depths.append(depth_value)
    #                 return True
    #     except ValueError:
    #         pass
    #     return False
    
    # def parse_log_file(self, filepath):
    #     """Parse a single log file"""
    #     print(f"Parsing log file: {filepath}")
        
    #     with open(filepath, 'r') as f:
    #         for line_num, line in enumerate(f, 1):
    #             line = line.strip()
    #             if not line:
    #                 continue
                
    #             # Try to parse sensor data first
    #             sensor_match = re.search(self.sensor_pattern, line)
    #             if sensor_match:
    #                 timestamp = self.parse_timestamp(sensor_match.group(1))
    #                 data_str = sensor_match.group(2)
    #                 self.parse_sensor_line(timestamp, data_str)
    #                 continue
                
    #             # Try to parse motor data
    #             if 'input:' in line:
    #                 self.parse_motor_line(line)
    #                 continue
                
    #             # Try to parse FPS data
    #             if 'fps:' in line:
    #                 self.parse_fps_line(line)
    #                 continue
                
    #             # Try to parse standalone depth values
    #             self.parse_depth_line(line)
        
    #     print(f"Parsed {len(self.sensor_data)} sensor entries")
    #     print(f"Parsed {len(self.motor_inputs)} motor input entries") 
    #     print(f"Parsed {len(self.fps_data)} FPS entries")
    #     print(f"Parsed {len(self.stereo_depths)} depth entries")
    
    # def synchronize_data(self):
    #     """Synchronize data streams to common time base"""
    #     if not self.sensor_data:
    #         print("Warning: No sensor data found for synchronization")
    #         return
        
    #     # Use sensor data as primary time reference
    #     sensor_timestamps = np.array([entry['timestamp'] for entry in self.sensor_data])
        
    #     # Start time for all data
    #     start_time = sensor_timestamps[0] if len(sensor_timestamps) > 0 else 0
        
    #     # Create time vector
    #     self.timestamps = sensor_timestamps - start_time
        
    #     print(f"Data synchronized. Mission duration: {self.timestamps[-1]:.2f} seconds")
    
    # def create_npz_data(self):
    #     """Convert parsed data to NPZ format matching original structure"""
    #     if not self.sensor_data:
    #         print("Error: No data to convert")
    #         return None
        
    #     n_samples = len(self.sensor_data)
        
    #     # Extract sensor data arrays
    #     quat_data = np.array([entry['quat'] for entry in self.sensor_data])
    #     acc_data = np.array([entry['acc'] for entry in self.sensor_data]) 
    #     gyr_data = np.array([entry['gyr'] for entry in self.sensor_data])
    #     depth_data = np.array([entry['depth'] for entry in self.sensor_data])
    #     altitude_data = np.array([entry['altitude'] for entry in self.sensor_data])
    #     voltage_data = np.array([entry['voltage'] for entry in self.sensor_data])
        
    #     # Handle motor data - pad or truncate to match sensor data length
    #     if self.motor_inputs:
    #         motor_array = np.array(self.motor_inputs)
    #         if len(motor_array) >= n_samples:
    #             u_data = motor_array[:n_samples]
    #             input_data = motor_array[:n_samples]
    #         else:
    #             # Pad with last values if motor data is shorter
    #             u_data = np.zeros((n_samples, 10))
    #             input_data = np.zeros((n_samples, 10))
    #             u_data[:len(motor_array)] = motor_array
    #             input_data[:len(motor_array)] = motor_array
    #             # Fill remaining with last valid values
    #             if len(motor_array) > 0:
    #                 u_data[len(motor_array):] = motor_array[-1]
    #                 input_data[len(motor_array):] = motor_array[-1]
    #     else:
    #         u_data = np.zeros((n_samples, 10))
    #         input_data = np.zeros((n_samples, 10))
        
    #     # Handle stereo depth data
    #     if self.stereo_depths:
    #         stereo_array = np.array(self.stereo_depths)
    #         if len(stereo_array) >= n_samples:
    #             stereo_depth_data = stereo_array[:n_samples]
    #         else:
    #             stereo_depth_data = np.zeros(n_samples)
    #             stereo_depth_data[:len(stereo_array)] = stereo_array
    #     else:
    #         stereo_depth_data = np.zeros(n_samples)
        
    #     # Create synthetic data for missing fields (marked as estimated)
    #     # These would normally come from your control system
    #     q_data = np.zeros((n_samples, 10))  # Joint positions - would need actual data
    #     dq_data = np.zeros((n_samples, 10))  # Joint velocities - would need actual data  
    #     qd_data = np.zeros((n_samples, 10))  # Desired joint positions - would need actual data
    #     dqd_data = np.zeros((n_samples, 10))  # Desired joint velocities - would need actual data
    #     nav_u_data = np.zeros((n_samples, 4))  # Navigation control - would need actual data
    #     depth_d_data = np.zeros(n_samples)  # Desired depth - would need actual data
    #     yaw_d_data = np.zeros(n_samples)  # Desired yaw - would need actual data
    #     alt_data = altitude_data[:, 0] if altitude_data.shape[1] > 0 else np.zeros(n_samples)
    #     stereo_point_data = np.zeros((2, n_samples))  # Stereo points - would need actual data
        
    #     # Create NPZ data dictionary
    #     npz_data = {
    #         'q': q_data,                    # Joint positions (NEEDS REAL DATA)
    #         'dq': dq_data,                  # Joint velocities (NEEDS REAL DATA)
    #         't': self.timestamps,           # Time vector
    #         'input': input_data,            # Motor inputs ✓
    #         'u': u_data,                    # Control inputs ✓
    #         'nav_u': nav_u_data,           # Navigation control (NEEDS REAL DATA)
    #         'qd': qd_data,                 # Desired positions (NEEDS REAL DATA)  
    #         'dqd': dqd_data,               # Desired velocities (NEEDS REAL DATA)
    #         'depth': depth_data,           # Depth sensor ✓
    #         'depth_d': depth_d_data,       # Desired depth (NEEDS REAL DATA)
    #         'quat': quat_data,             # Quaternion orientation ✓
    #         'alt': alt_data,               # Altitude ✓
    #         'yaw_d': yaw_d_data,          # Desired yaw (NEEDS REAL DATA)
    #         'stereo_depth': stereo_depth_data,  # Stereo depth ✓
    #         'stereo_point': stereo_point_data,  # Stereo points (NEEDS REAL DATA)
    #         # Additional logged data
    #         'acc': acc_data,               # Accelerometer ✓
    #         'gyr': gyr_data,              # Gyroscope ✓  
    #         'voltage': voltage_data,       # Battery voltage ✓
    #         'fps': np.array(self.fps_data) if self.fps_data else np.array([])  # Camera FPS ✓
    #     }
        
    #     return npz_data
    
    # def save_npz(self, output_path, data=None):
    #     """Save data to NPZ file"""
    #     if data is None:
    #         data = self.create_npz_data()
        
    #     if data is None:
    #         print("Error: No data to save")
    #         return False
        
    #     np.savez_compressed(output_path, **data)
    #     print(f"Data saved to: {output_path}")
        
    #     # Print summary
    #     print(f"\nData Summary:")
    #     for key, value in data.items():
    #         if isinstance(value, np.ndarray):
    #             print(f"{key}: shape {value.shape}")
        
    #     return True

# def main():
#     parser = argparse.ArgumentParser(description='Convert turtle robot log files to NPZ format')
#     parser.add_argument('log_files', nargs='+', help='Path to log files')
#     parser.add_argument('-o', '--output', help='Output NPZ file path')
    
#     args = parser.parse_args()
    
#     # Create parser instance
#     log_parser = TurtleLogParser()
    
#     # Parse all provided log files
#     for log_file in args.log_files:
#         if os.path.exists(log_file):
#             log_parser.parse_log_file(log_file)
#         else:
#             print(f"Warning: Log file not found: {log_file}")
    
#     if not log_parser.sensor_data and not log_parser.motor_inputs:
#         print("Error: No valid data found in log files")
#         return
    
#     # Synchronize data streams
#     log_parser.synchronize_data()
    
#     # Determine output path
#     if args.output:
#         output_path = args.output
#     else:
#         # Generate output path based on first log file
#         base_name = os.path.splitext(os.path.basename(args.log_files[0]))[0]
#         output_path = f"{base_name}_converted.npz"
    
#     # Save to NPZ
#     success = log_parser.save_npz(output_path)
    
#     if success:
#         print(f"\n✓ Successfully converted logs to NPZ format!")
#         print(f"Use this file with your existing visualization code.")
#         print(f"\nNote: Some fields are filled with zeros and marked as 'NEEDS REAL DATA'")
#         print(f"You may need to modify the parser to extract these from your specific logs.")

# if __name__ == "__main__":
#     # Example usage if run directly
#     if len(os.sys.argv) == 1:
#         print("Usage: python log_parser.py <log_file1> [log_file2] ... [-o output.npz]")
#         print("\nExample:")
#         print("python log_parser.py robot_log1.log robot_log2.log -o trial_data.npz")
#     else:
#         main()