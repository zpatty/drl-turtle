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

class TurtleLogger(Node):
    """
    This node is responsible for logging all data collected from the turtle's sensors and motors.
    ** This node should only be run on a non-turtle machine!
    """

    def __init__(self):
        super().__init__('logger_node')
        self.declare_parameter('port', descriptor = ParameterDescriptor(type = ParameterType.PARAMETER_STRING_ARRAY))

        self.freq = 100
        self.t_0 = 0

        self.n_axis = 3
        self.acc_data = np.zeros((self.n_axis,1))
        self.gyr_data = np.zeros((self.n_axis,1))
        self.mag_data = np.zeros((self.n_axis,1))
        self.timestamps = np.zeros((1,1))

        timer_period =  1/self.freq  #seconds

        self.sensor_msg = TurtleSensors()
        
        self.sensors_publisher = self.create_publisher(TurtleSensors, 'turtle_sensors', 10)
        # self.xiao = serial.Serial('/dev/ttyACM0', 115200, timeout=3)   
        self.timer = self.create_timer(timer_period, self.handle_sensors)

    def mode_callback(self, msg):
        """
        Callback that checks what mode the turtle is in.
        """    
        pass

    def sensor_callback(self, msg):
        """
        log sensors
        """
        pass

    def motor_callback(self, msg):
        """
        log motor pos
        """
        pass
        
def main(args=None):
    rclpy.init(args=args)

    sensors_node = TurtleLogger()

    rclpy.spin(sensors_node)

    sensors_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()