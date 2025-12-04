import numpy as np
import re
import json
import os
from datetime import datetime
import argparse

class TurtleLogParser:
    """
    Parser for turtle robot log files to extract sensor data, motor commands,
    navigation commands, and other relevant information. Converts the parsed data
    into a structured NPZ format compatible with existing analysis tools.
    """
    def __init__(self):
        self.sensor_data = []
        self.motor_inputs = []
        self.stereo_depths = []
        self.headings = []
        self.fps_data = []
        self.timestamps_sensor_node = []
        self.timestamps_logger_node = []
        self.nav_commands = []
        self.stereo_debug_depths = []
        self.turn_commands = []
        self.turn_command = None
        self.stereo_count = 0
        # Regex patterns for different log types
        self.sensor_pattern = r'\[INFO\] \[(\d+\.\d+)\] \[turtle_sensors_node\]: (.+)'
        self.motor_pattern = r'input: (\[.+\])'
        self.fps_pattern = r'fps: ([\d.]+)'
        self.heading_pattern = r'heading: ([-\d.]+)'
        self.stereo_debug_pattern = r'\[DEBUG\] stereo depth: ([\d.]+)'
        self.logger_pattern = r'\[INFO\] \[(\d+\.\d+)\] \[log_node\]: (.+)'
        self.turn_pattern = r'\[DEBUG\] (turn \w+)'
        self.count_turn_commands = 0
        self.no_flag_u_count = 0
        self.depth_value = None
        self.flag_turn = None
        # to regenerate yaw_d
        self.turn_counts = []
        self.yaw_d = 2.2
        self.u_last = [1, 0, 0, 0, 0]
        self.count = 0
        self.yaw_ds = [self.yaw_d]
        self.depth_pattern = r'^([\d.]+)$'  # Single depth values
        
    def parse_timestamp(self, timestamp_str):
        """Convert ROS timestamp to seconds"""
        return float(timestamp_str)
    
    def parse_logger_line(self, timestamp):
        """Parses ROS timestamp from logger node"""
        self.timestamps_logger_node.append(timestamp)
    
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

    def parse_sensor_line(self, timestamp, data_str):
        """Parse sensor JSON data"""
        try:
            # Clean up the data string and parse as JSON
            if data_str.strip().startswith('{') and data_str.strip().endswith('}'):
                sensor_json = json.loads(data_str)
                entry = {
                    'timestamp': timestamp,
                    'quat': sensor_json.get('Quat', [np.nan, np.nan, np.nan, np.nan]),
                    'acc': sensor_json.get('Acc', [np.nan, np.nan, np.nan]),
                    'gyr': sensor_json.get('Gyr', [np.nan, np.nan, np.nan]),
                    'depth': sensor_json.get('Depth', [np.nan])[0] if sensor_json.get('Depth') else np.nan,
                    'altitude': sensor_json.get('Altitude', [np.nan, np.nan]),
                    'voltage': sensor_json.get('Voltage', [np.nan])[0] if sensor_json.get('Voltage') else np.nan
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
        """Parse standalone depth values, specifically from Sub_Cam_ut.py"""
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
        """Parse navigation control commands
            updates navu and desired yaw_d if depth is None
        """
        self.nav_u_pattern = r'no flag u: (\[.+\])'

        try:
            nav_match = re.search(self.nav_u_pattern, line)
            if nav_match:
                nav_str = nav_match.group(1)
                # Parse the array string
                nav_values = eval(nav_str)  # Safe here since we control the format
                # print(f"parse nav u before: {self.no_flag_u_count}, len yaw: {len(self.yaw_ds)}, nav u: {len(self.nav_commands)}, depth_value: {self.depth_value}, flag_turn: {self.flag_turn}, turn_command: {self.turn_command}")

                if len(nav_values) >= 4: 
                    # also add yaw_d here
                    if self.depth_value is not None:
                        if not self.flag_turn:
                            if self.depth_value < 3.5:
                                u_yaw = nav_values[3]
                                if u_yaw < 0:
                                    yaw_d = np.arctan2(np.sin(self.heading + np.deg2rad(120)), np.cos(self.heading + np.deg2rad(120)))
                                elif u_yaw > 0:
                                    yaw_d = np.arctan2(np.sin(self.heading - np.deg2rad(120)), np.cos(self.heading - np.deg2rad(120)))
                                else:
                                    print("we don't have u yaw for some reason")
                                self.yaw_ds.append(yaw_d)
                                self.nav_commands.append(nav_values[:4])  # Take first 4 for nav_u
                                self.no_flag_u_count += 1    
                            else:
                                # if depth > 3.5, just repeat last yaw_d and the current no flag u
                                # print(f"--------------depth is {self.depth_value}, repeating last yaw_d")
                                yaw_d = self.yaw_ds[-1]
                                self.yaw_ds.append(yaw_d)
                                self.nav_commands.append(nav_values[:4])  # Take first 4 for nav_u
                                self.no_flag_u_count += 1                        
                            # print(f"we are appending: {self.no_flag_u_count}, len yaw: {len(self.yaw_ds)}, nav u: {len(self.nav_commands)}, depth_value: {self.depth_value}, flag_turn: {self.flag_turn}, turn_command: {self.turn_command}")
                        else:
                            print("--------------we are in turn, repeating last yaw_d")
                    else:
                        # if no depth value, just repeat last yaw_d and the current no flag u
                        print("--------------no depth value, repeating last yaw_d")
                        yaw_d = self.yaw_ds[-1]
                        self.yaw_ds.append(yaw_d)
                        self.nav_commands.append(nav_values[:4])  # Take first 4 for nav_u
                        self.no_flag_u_count += 1

                    # print(f"parse nav u after: {self.no_flag_u_count}, len yaw: {len(self.yaw_ds)}, nav u: {len(self.nav_commands)}, depth_value: {self.depth_value}, flag_turn: {self.flag_turn}, turn_command: {self.turn_command}")
                    return True
                else:
                    print(f"-------------nav u length not 4, got {len(nav_values)}")
            else:
                print("-------------Failed to parse nav u line because of no match")
        except (SyntaxError, ValueError):
            print("-------------Failed to parse nav u line due to syntax error")
            pass
    
    def parse_stereo_debug_line(self, line):
        """Parse stereo debug depth values from untethered planning node"""
        # print(f"stereo count: {self.stereo_count}, {self.count_turn_commands}, len yaw: {len(self.yaw_ds)}, nav u: {len(self.nav_commands)}, depth_value: {self.depth_value}, flag_turn: {self.flag_turn}, turn_command: {self.turn_command}")
        self.stereo_count += 1
        try:
            stereo_match = re.search(self.stereo_debug_pattern, line)
            if stereo_match:

                depth_value = float(stereo_match.group(1))
                self.stereo_debug_depths.append(depth_value)
                self.depth_value = depth_value
                # print(f"stereo match!: {self.stereo_count}, {self.count_turn_commands}, len yaw: {len(self.yaw_ds)}, nav u: {len(self.nav_commands)}, depth_value: {self.depth_value}, flag_turn: {self.flag_turn}, turn_command: {self.turn_command}")

                if self.flag_turn:
                    self.turn_commands.append(self.turn_command)
                    u = self.nav_commands[-1].copy()
                    if self.turn_command == "turn left":
                        u_yaw = -1.0
                        u_roll = -1.0
                        u[3] = u_yaw
                        u[1] = u_roll
                    else:
                        u_yaw = 1.0
                        u_roll = 1.0
                        u[3] = u_yaw
                        u[1] = u_roll
                    self.nav_commands.append(u) 
                    self.u_last = u
                    # should pass the same desired yaw_d as before
                    yaw_d = self.yaw_ds[-1].copy()
                    self.yaw_ds.append(yaw_d)
                    self.count_turn_commands += 1
                    self.count += 1
                else:
                    self.turn_commands.append("no turn")

                    # print("-----------------not in turn, just append from no flag u")

                
                if self.depth_value > 3.0 and self.flag_turn:
                    self.flag_turn = False
                    # self.nav_commands.append(self.u_last) 
                    # self.yaw_ds.append(self.yaw_ds[-1].copy())
                    self.count += 1

                # print(f"count: {self.count}, depth: {self.depth_value}, turn command: {self.turn_command}, flag_turn: {self.flag_turn}, yaw_d: {self.yaw_ds[-1]}, nav_u: {self.nav_commands[-1]}")
                return True
            else:
                # print("-------------Failed to parse stereo debug depth")
                # print(f"line: {line}")
                # case where we probably got None, in which case handle it with nan
                if 'None' in line:
                    print("-------------We got None")
                    self.stereo_debug_depths.append(np.nan)
                    self.depth_value = None
                    self.turn_commands.append("no turn")
                    # print(f"None case stereo count: {self.stereo_count}, {self.count_turn_commands}, len yaw: {len(self.yaw_ds)}, nav u: {len(self.nav_commands)}, depth_value: {self.depth_value}, flag_turn: {self.flag_turn}, turn_command: {self.turn_command}")
                    return True    
            return False                
        except ValueError:
            print("-------------Failed to parse stereo debug depth")
            pass
        return False
    
    def parse_turn_command_line(self, line):
        """Parse turn commands from untethered planning node, used to figure out nav u direction in logging"""
        try:
            turn_match = re.search(self.turn_pattern, line)
            if turn_match:
                turn_cmd = turn_match.group(1)
                print(f"count turn commands: {self.count_turn_commands}")
                self.turn_counts.append(self.count_turn_commands)
                self.count_turn_commands = 0
                # self.turn_commands.append(turn_cmd)
                self.turn_command = turn_cmd
                # print(f"turn command: {self.turn_command}")
                # remember to pop out last nav command and replace with turn command
                if len(self.nav_commands) > 0:
                    u = self.nav_commands.pop()
                    # print("---------------------popped last nav command for turn:", u)
                self.flag_turn = True
                if turn_cmd == "turn right":
                    u_yaw = 1.0
                    u_roll = 1.0
                    u[3] = u_yaw
                    u[1] = u_roll
                else:  # left turn
                    u_yaw = -1.0
                    u_roll = -1.0
                    u[3] = u_yaw
                    u[1] = u_roll
                self.nav_commands.append(u) 
                self.u_last = u
                # should pass the same desired yaw_d as before
                # yaw_d = self.yaw_ds[-1].copy()
                # self.yaw_ds.append(yaw_d)
                return True
            else:
                print("----------------------Failed to parse turn command because of no match")
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
                # Try to parse logger timestamp
                if 'update' in line:
                    logger_match = re.search(self.logger_pattern, line)
                    if logger_match:
                        logger_timestamp = self.parse_timestamp(logger_match.group(1))
                        # print(f"logger_timestamp: {logger_timestamp}\n")
                        data_str = logger_match.group(2)
                        self.parse_logger_line(logger_timestamp)
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
                                
                # Try to parse stereo debug depth
                if '[DEBUG] stereo depth:' in line:
                    self.parse_stereo_debug_line(line)
                    continue

                # Try to parse navigation commands
                if 'no flag u:' in line:
                    self.parse_nav_u_line(line)
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
        print(f"Parsed {len(self.stereo_debug_depths)} stereo debug depth entries")
        print(f"Parsed {len(self.nav_commands)} navigation command entries")
        print(f"Parsed {len(self.turn_commands)} turn command entries")
        print(f"Parsed {len(self.timestamps_logger_node)} logger timestamps")

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
            self.timestamps_sensor_node = sensor_timestamps - start_time
            self.primary_length = len(self.sensor_data)
        else:
            # Create synthetic timestamps based on largest dataset
            self.primary_length = max_length
            # Assume ~10Hz sampling rate for navigation data
            self.timestamps_sensor_node = np.arange(max_length) * 0.1
            start_time = 0
        
        print(f"Data synchronized. Primary length: {self.primary_length}, Duration: {self.timestamps_sensor_node[-1] if len(self.timestamps_sensor_node) > 0 else 0:.2f} seconds")
    
    def create_npz_data(self):
        """Convert parsed data to NPZ format matching original structure"""
        # if self.primary_length == 0:
        #     print("Error: No data to convert")
        #     return None
        self.primary_length = len(self.sensor_data)
        n_samples = self.primary_length
        
        # handle logger data too: 
        if len(self.timestamps_logger_node) > 0:
            logger_start_time = self.timestamps_logger_node[0]
            self.timestamps_logger_node = np.array(self.timestamps_logger_node) - logger_start_time
        else:
            self.timestamps_logger_node = np.zeros(n_samples)
        # Handle sensor data if available
        if self.sensor_data:
            quat_data = np.array([entry['quat'] for entry in self.sensor_data])
            acc_data = np.array([entry['acc'] for entry in self.sensor_data]) 
            gyr_data = np.array([entry['gyr'] for entry in self.sensor_data])
            depth_data = np.array([entry['depth'] for entry in self.sensor_data])
            altitude_data = np.array([entry['altitude'] for entry in self.sensor_data])
            voltage_data = np.array([entry['voltage'] for entry in self.sensor_data])
            sensor_timestamps = np.array([entry['timestamp'] for entry in self.sensor_data])
            start_time = sensor_timestamps[0]
            self.timestamps_sensor_node = sensor_timestamps - start_time
            
        else:
            print("placing zeros for sensor data")
            # Create zeros if no sensor data
            quat_data = np.zeros((n_samples, 4))
            acc_data = np.zeros((n_samples, 3))
            gyr_data = np.zeros((n_samples, 3))
            depth_data = np.zeros(n_samples)
            altitude_data = np.zeros((n_samples, 2))
            voltage_data = np.zeros(n_samples)
        
        if self.nav_commands:
            nav_array = np.array(self.nav_commands)
            nav_u_data = nav_array
        else:
            print("placing zeros for nav u data")
            nav_u_data = np.zeros((n_samples, 4))
        
        # Handle stereo depth data
        stereo_depth_data = np.zeros(n_samples)
        if self.stereo_debug_depths:
            stereo_debug_array = np.array(self.stereo_debug_depths)
            stereo_debug_depth_data = stereo_debug_array
        if self.stereo_depths:  # Fallback to regular stereo depths
            stereo_array = np.array(self.stereo_depths)
            min_len = min(len(stereo_array), n_samples)
            stereo_depth_data = stereo_array
        
        # Handle motor data
        if self.motor_inputs:
            motor_array = np.array(self.motor_inputs)
            input_data = motor_array
        else:
            print("placing zeros for u data and input data")
            u_data = np.zeros((n_samples, 10))
            input_data = np.zeros((n_samples, 10))
        
        # Create synthetic data for missing fields
        q_data = np.zeros((n_samples, 10))  # Joint positions - would need actual data
        dq_data = np.zeros((n_samples, 10))  # Joint velocities - would need actual data  
        qd_data = np.zeros((n_samples, 10))  # Desired joint positions - would need actual data
        dqd_data = np.zeros((n_samples, 10))  # Desired joint velocities - would need actual data
        depth_d_data = np.zeros(n_samples)  # Desired depth - would need actual data
        print(f"len of yaw_ds: {len(self.yaw_ds)}, expected: 8422")
        yaw_d_data = np.array(self.yaw_ds)  # Desired yaw from untethered planner 
        alt_data = altitude_data[:, 0] if altitude_data.shape[1] > 0 else np.zeros(n_samples)
        stereo_point_data = np.zeros((2, n_samples))  # Stereo points - would need actual data
        # print(f"we have {len(self.turn_commands)} turn commands! {self.turn_commands}")
        print(f"we have {len(self.turn_commands)} turn commands!")
        print(f"we have {self.count_turn_commands} yaw d repeats!")
        print(f"we have {self.no_flag_u_count} no flag u commands!")
        print(f"we have {sum(self.turn_counts)} turn counts!")
        # Create NPZ data dictionary
        npz_data = {
            'q': q_data,                    # Joint positions (NEEDS REAL DATA)
            'dq': dq_data,                  # Joint velocities (NEEDS REAL DATA)
            't_sensor_node': self.timestamps_sensor_node,           # Time vector from TurtleSensors.py ✓
            't_logger_node': self.timestamps_logger_node,           # Time vector from logger.py
            'u': input_data,            # Motor inputs current ✓
            'nav_u': nav_u_data,            # Navigation u ✓
            'qd': qd_data,                  # Desired positions (NEEDS REAL DATA)  
            'dqd': dqd_data,                # Desired velocities (NEEDS REAL DATA)
            'depth': depth_data,            # Depth sensor ✓
            'depth_d': depth_d_data,        # Desired depth (NEEDS REAL DATA)
            'quat': quat_data,              # Quaternion orientation ✓
            'alt': alt_data,                # Altitude ✓
            'yaw_d': yaw_d_data,            # Desired yaw ✓ (from headings)
            'stereo_depth': stereo_depth_data,  # Stereo depth from Sub Cam ✓
            'stereo_debug_depth': stereo_debug_depth_data,   # Stereo depth from untethered planning node
            'stereo_point': stereo_point_data,  # Stereo points (NEEDS REAL DATA)
            'acc': acc_data,                # Accelerometer ✓
            'gyr': gyr_data,                # Gyroscope ✓  
            'voltage': voltage_data,        # Battery voltage ✓
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
        
        print(f"final count: {self.count}")
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
    # log_parser.synchronize_data()
    
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