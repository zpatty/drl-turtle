import serial
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from std_msgs.msg import String
from turtle_interfaces.msg import TurtleTraj, TurtleSensors, TurtleMotorPos
import numpy as np
import json
import sys
import os
submodule = os.path.expanduser("~") + "/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware"
sys.path.append(submodule)

class TurtleSensorsPublisher(Node):
    """
    This node is responsible for handling the arduino to serial communication of all sensor readouts. 
    Responsible for periodically reading from seeeduino on the turtle and publishing all sensor data to the motors and keyboard nodes. 
    Ex. it reads IMU data, packages it into custom turtle msg, and sends it to motors to carry out some action. 

    This node constantly sends voltage data to the motor node
    The motor node is what handles the voltage < threshold

    Once the motor node finds that most recent voltage < threshold, it will collect all trajectory data, publish it to the sensors node,
    and then the sensors node will be responsible for packaging everything together 

    Perhaps find a way to have both nodes running concurrently?
    the only issue is the ability to interrupt a process in callback, since you need to be able to consistently interrupt,
    which goes back to our issue of being able to update variables while also being able to run entire trajectories without stalling. 
    """

    def __init__(self, topic):
        super().__init__('sensors_node')
        self.freq = 100
        self.t_0 = 0
        self.n_axis = 3
        self.sensor_msg = TurtleSensors()
        self.sensor_mode = 'rest'
        # subscribes to keyboard setting different turtle modes 
        self.mode_cmd_sub = self.create_subscription(
            String,
            topic,
            self.sensor_mode_callback,
            10)
        self.sensors_publisher = self.create_publisher(TurtleSensors, topic, 10)
        self.xiao = serial.Serial('/dev/ttyACM0', 115200, timeout=3)   
        self.xiao.reset_input_buffer()

        self.acc_data = np.zeros((self.n_axis,1))
        self.gyr_data = np.zeros((self.n_axis,1))
        self.quat_data = np.zeros((4,1))
        self.tau_data = np.zeros((1,1))
        self.voltage_data = np.zeros((1,1))
        self.voltage = 12

    def sensor_mode_callback(self, msg):
        """
        Callback function that updates the mode the turtle should be in.
        This method is what enables us to set "emergency stops" mid-trajectory. 
        """
        # global mode
        if msg.data == 'traj1':
            self.mode = 'traj1'
        elif msg.data == 'traj2':
            self.mode = 'traj2'
        elif msg.data == 'stop':
            self.mode = 'stop'
        else:
            self.mode = 'rest'

    def read_voltage(self):
        turtle_msg = TurtleSensors()
        sensors = self.xiao.readline()
        if len(sensors) != 0:
        # this ensures the right json string format
            if sensors[0] == 32 and sensors[-1] == 10:
                try:
                    sensor_dict = json.loads(sensors.decode('utf-8'))
                except:
                    no_check = True
                # add sensor data
                if no_check == False:
                    sensor_keys = ('Voltage')
                    if set(sensor_keys).issubset(sensor_dict):
                        volt = sensor_dict['Voltage'][0]
                        self.voltage = volt   
                        turtle_msg.voltage = [self.voltage]   
                        self.voltage_pub.publish(turtle_msg)



    def handle_sensors(self):
        # msg.header.stamp = self.get_clock().now().to_msg()
        sensors = self.xiao.readline()
        # sensors = xiao.readline().decode('utf-8').rstrip()
        # print(f"raw: {sensors}\n")
        # # print(f"raw first character: {sensors[0]}")
        if len(sensors) != 0:
        # this ensures the right json string format
            if sensors[0] == 32 and sensors[-1] == 10:
                try:
                    sensor_dict = json.loads(sensors.decode('utf-8'))
                except:
                    no_check = True
                # add sensor data
                if no_check == False:
                    sensor_keys = ('Acc', 'Gyr', 'Quat', 'Voltage')
                    if set(sensor_keys).issubset(sensor_dict):
                        acc = np.array(sensor_dict['Acc']).reshape((3,1))
                        gyr = np.array(sensor_dict['Gyr']).reshape((3,1))
                        quat = np.array(sensor_dict['Quat']).reshape((4,1))
                        volt = sensor_dict['Voltage'][0]
                        self.acc_data = np.append(self.acc_data, acc, axis=1)
                        self.gyr_data = np.append(self.gyr_data, gyr, axis=1)
                        self.quat_data = np.append(self.quat_data, quat, axis=1)
                        self.voltage_data = np.append(self.voltage_data, volt)
                        self.voltage = volt    

def main(args=None):
    rclpy.init(args=args)
    sensors_node = TurtleSensorsPublisher('turtle_sensors')

    try: 
        while rclpy.ok():
            rclpy.spin_once(sensors_node)

            if sensors_node.mode == 'stop':
                print("ending entire program...")
                sensors_node.read_voltage()
                print("stop received but still in rest mode....")
            elif sensors_node.mode == 'rest':
                sensors_node.read_voltage()
                print("rest mode....")
                print(f"current voltage: {sensors_node.voltage}\n")
            elif sensors_node.mode == 'traj_input':
                print("starting sensor recording")
                while 1:
                    rclpy.spin_once(sensors_node)
                    sensors_node.handle_sensors()
                    if sensors_node.mode == 'rest' or sensors_node.mode == 'stop':
                        print("saving data...")
                        sensors_node.publish_turtle_data() 
                        break
            else:
                print("wrong command received....")
    except Exception as e:
        print("some error occurred")
        exec_type, obj, tb = sys.exc_info()
        fname = os.path.split(tb.tb_frame.f_code.co_filename)[1]
        print(exec_type, fname, tb.tb_lineno)
        print(e)
        


        
    
    rclpy.shutdown()

if __name__ == '__main__':
    main()

