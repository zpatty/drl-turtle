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
        self.mode = 'rest'
        # subscribes to keyboard setting different turtle modes 
        self.mode_cmd_sub = self.create_subscription(
            String,
            topic,
            self.sensor_mode_callback,
            10)
        self.sensors_pub = self.create_publisher(TurtleSensors, 'turtle_sensors', 10)
        self.cmd_received_pub = self.create_publisher(
            String,
            'turtle_state',
            10
        )
        self.xiao = serial.Serial('/dev/ttyACM0', 115200, timeout=3)   
        self.xiao.reset_input_buffer()
        self.create_rate(50)
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
        print(f"MESSAGE : {msg.data}")
        if msg.data == 'traj1':
            self.mode = 'traj1'
        elif msg.data == 'traj2':
            self.mode = 'traj2'
        elif msg.data == 'teacher':
            self.mode = 'teacher'
        elif msg.data == 'stop':
            self.mode = 'stop'
        else:
            self.mode = 'rest'

    def read_voltage(self):
        no_check = False
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
                    sensor_keys = {'Voltage'}
                    if set(sensor_keys).issubset(sensor_dict):
                        volt = sensor_dict['Voltage'][0]
                        self.voltage = volt   
                        turtle_msg.voltage = [self.voltage]   
                        # self.voltage_pub.publish(turtle_msg)



    def handle_sensors(self):
        no_check = False
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
    
    def publish_turtle_data(self):
        turtle_msg = TurtleSensors()
        
        # # extract and flatten numpy arrays
        # print("quat")
        quat_x = self.quat_data[0, :]
        quat_y = self.quat_data[1, :]
        quat_z = self.quat_data[2, :]
        quat_w = self.quat_data[3, :]

        # print("acc")
        acc_x = self.acc_data[0, :]
        acc_y = self.acc_data[1, :]
        acc_z = self.acc_data[2, :]

        # print("gyr")
        gyr_x = self.gyr_data[0, :]
        gyr_y = self.gyr_data[1, :]
        gyr_z = self.gyr_data[2, :]

        # print("wuat msg")
        # # quaternion 
        # print(f"quat x: {quat_x}\n")
        quatx = quat_x.tolist()
        # print(f"quat x: {quatx}\n")
        # print(f"element type: {type(quatx[0])}")
        turtle_msg.imu.quat_x = quat_x.tolist()
        turtle_msg.imu.quat_y = quat_y.tolist()
        turtle_msg.imu.quat_z = quat_z.tolist()
        turtle_msg.imu.quat_w = quat_w.tolist()
        # print("gyr msg")
        # # angular velocity
        turtle_msg.imu.gyr_x = gyr_x.tolist()
        turtle_msg.imu.gyr_y = gyr_y.tolist()
        turtle_msg.imu.gyr_z = gyr_z.tolist()
        # print("acc msg")
        # # linear acceleration
        turtle_msg.imu.acc_x = acc_x.tolist()
        turtle_msg.imu.acc_y = acc_y.tolist()
        turtle_msg.imu.acc_z = acc_z.tolist()
        
        # print("volt msg")
        # # voltage
        volt_data = self.voltage_data.tolist()
        # print(f"voltage data: {volt_data}")
        turtle_msg.voltage = volt_data  

        # timestamps and t_0
        # turtle_msg.timestamps = timestamps.tolist()
        # turtle_msg.t_0 = t_0

        # motor positions, desired positions and tau data
        # turtle_msg.q = self.np2msg(q_data) 
        # turtle_msg.dq = self.np2msg(dq_data)
        # TODO: report dq?
        # qd_squeezed = self.np2msg(self.qds)
        # print(f"checking type: {type(qd_squeezed)}")
        # for i in qd_squeezed:
        #     print(f"check element type: {type(i)}")
        # turtle_msg.qd = qd_squeezed
        # turtle_msg.tau = self.np2msg(tau_data)

        # publish msg 
        self.sensors_pub.publish(turtle_msg)

        # reset the sensor variables for next trajectory recording
        # self.qds = np.zeros((10,1))
        # self.dqds = np.zeros((10,1))
        # self.ddqds = np.zeros((10,1))
        # self.tvec = np.zeros((1,1))
        self.acc_data = np.zeros((self.n_axis,1))
        self.gyr_data = np.zeros((self.n_axis,1))
        self.mag_data = np.zeros((self.n_axis,1))
        self.voltage_data = np.zeros((1,1))

def main(args=None):
    rclpy.init(args=args)
    print("init sensors node")
    sensors_node = TurtleSensorsPublisher('turtle_mode_cmd')
    print("it has been made")
    try: 
        while rclpy.ok():
            rclpy.spin_once(sensors_node)
            if sensors_node.mode == 'stop':
                print("ending entire program...")
                # sensors_node.read_voltage()
                cmd_msg = String()
                cmd_msg.data = "stop_received"
                sensors_node.cmd_received_pub.publish(cmd_msg)
                break
                print("stop received but still in rest mode....")
            elif sensors_node.mode == 'rest':
                sensors_node.read_voltage()
                print("rest mode....")
                print(f"current voltage: {sensors_node.voltage}\n")
            elif sensors_node.mode == 'teacher':
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

