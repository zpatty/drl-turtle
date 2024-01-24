import serial
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from turtle_interfaces.msg import TurtleSensors
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
    """

    def __init__(self, topic):
        super().__init__('sensors_node')
        self.declare_parameter('port', descriptor = ParameterDescriptor(type = ParameterType.PARAMETER_STRING_ARRAY))

        # self.port = self.get_parameter('port').get_parameter_value().string_array_value
        # self.baud = self.get_parameter('baud').get_parameter_value().string_array_value
        self.freq = 100
        self.t_0 = 0

        self.n_axis = 3
        self.acc_data = np.zeros((self.n_axis,1))
        self.gyr_data = np.zeros((self.n_axis,1))
        self.mag_data = np.zeros((self.n_axis,1))
        self.timestamps = np.zeros((1,1))

        timer_period =  1/self.freq  #seconds

        self.sensor_msg = TurtleSensors()
        
        self.sensors_publisher = self.create_publisher(TurtleSensors, topic, 10)
        # self.xiao = serial.Serial('/dev/ttyACM0', 115200, timeout=3)   
        self.timer = self.create_timer(timer_period, self.handle_sensors)

        
    def handle_sensors(self):
        # msg.header.stamp = self.get_clock().now().to_msg()

        # sensors = self.xiao.readline()
        # sensor_dict = json.loads(sensors.decode('utf-8'))

        # add time stamp
        t = 20
        time_elapsed = t-self.t_0
        # timestamps = np.append(timestamps, time_elapsed) 

        # add sensor data
        # acc_data = np.append(acc_data, sensor_dict['Acc'])
        # gyr_data = np.append(gyr_data, sensor_dict['Gyr'])
        # mag_data = np.append(mag_data, sensor_dict['Mag'])

        # pack it into message
        q = [0.0,1.0,2.0,3.0,4.0,5.0]

        # quaternion 
        self.sensor_msg.imu.orientation.x = q[0]
        self.sensor_msg.imu.orientation.y = q[1]
        self.sensor_msg.imu.orientation.z = q[2]
        self.sensor_msg.imu.orientation.w = q[3]

        # angular velocity
        self.sensor_msg.imu.angular_velocity.x = q[0]
        self.sensor_msg.imu.angular_velocity.y = q[0]
        self.sensor_msg.imu.angular_velocity.z = q[0]

        # linear acceleration
        self.sensor_msg.imu.linear_acceleration.x = q[0]
        self.sensor_msg.imu.linear_acceleration.y = q[0]
        self.sensor_msg.imu.linear_acceleration.z = q[0]

        # voltage
        self.sensor_msg.voltage = 12.0

        # publish msg 
        self.sensors_publisher.publish(self.sensor_msg)
        print(f"battery voltage: {self.sensor_msg.voltage}\n")


def main(args=None):
    rclpy.init(args=args)

    sensors_node = TurtleSensorsPublisher('turtle_sensors')

    rclpy.spin(sensors_node)

    sensors_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()