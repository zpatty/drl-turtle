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

        self.freq = 100
        self.t_0 = 0
        self.seq = 0

        self.n_axis = 3
        self.acc_data = np.zeros((self.n_axis,1))
        self.gyr_data = np.zeros((self.n_axis,1))
        self.quat_data = np.zeros((4,1))
        self.timestamps = np.zeros((1,1))

        timer_period =  1/self.freq  #seconds

        self.sensor_msg = TurtleSensors()
        
        self.sensors_publisher = self.create_publisher(TurtleSensors, topic, 10)
        self.xiao = serial.Serial('/dev/ttyACM0', 115200, timeout=3)   
        self.timer = self.create_timer(timer_period, self.handle_sensors)

    def handle_sensors(self):
        # msg.header.stamp = self.get_clock().now().to_msg()

        sensors = self.xiao.readline()
        # check that byte is valid string
        if sensors[0] == 123:
            sensor_dict = json.loads(sensors.decode('utf-8'))

            # add time stamp
            t = 20
            time_elapsed = t-self.t_0
            # timestamps = np.append(timestamps, time_elapsed) 

            # add sensor data
            acc_data = sensor_dict['Acc']
            gyr_data = sensor_dict['Gyr']
            quat_data = sensor_dict['Quat']

            # pack it into message
            # quaternion 
            self.sensor_msg.imu.orientation.x = quat_data[0]
            self.sensor_msg.imu.orientation.y = quat_data[1]
            self.sensor_msg.imu.orientation.z = quat_data[2]
            self.sensor_msg.imu.orientation.w = quat_data[3]

            # angular velocity
            self.sensor_msg.imu.angular_velocity.x = gyr_data[0]
            self.sensor_msg.imu.angular_velocity.y = gyr_data[1]
            self.sensor_msg.imu.angular_velocity.z = gyr_data[2]

            # linear acceleration
            self.sensor_msg.imu.linear_acceleration.x = acc_data[0]
            self.sensor_msg.imu.linear_acceleration.y = acc_data[1]
            self.sensor_msg.imu.linear_acceleration.z = acc_data[2]

            # voltage
            self.sensor_msg.voltage = sensor_dict['Voltage'][0]

            # # log data for safe keeping
            # # add sensor data
            # acc_data = np.append(acc_data, sensor_dict['Acc'])
            # gyr_data = np.append(gyr_data, sensor_dict['Gyr'])
            # mag_data = np.append(mag_data, sensor_dict['Mag'])

            # misc
            self.sensor_msg.header.stamp= rclpy.Time.now()
            self.sensor_msg.header.seq = self.seq
            self.seq = self.seq + 1
            # publish msg 
            self.sensors_publisher.publish(self.sensor_msg)
            print(f"battery voltage: {self.sensor_msg.voltage}\n")
    def reset_callback(self):
        """
        When we want to save data into rosbag and record a new one
        """
        self.seq = 0

def main(args=None):
    rclpy.init(args=args)

    sensors_node = TurtleSensorsPublisher('turtle_sensors')

    rclpy.spin(sensors_node)

    sensors_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()