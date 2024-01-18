import rclpy
from rclpy.node import Node
from rclpy.executors import Executor, MultiThreadedExecutor
from rcl_interfaces.msg import SetParametersResult
from turtle_interfaces.msg import TurtleTraj
import json
import sys
import os

submodule = os.path.expanduser("~") + "/drl-turtle/ros2_ws/src/py_pubsub/py_pubsub"
sys.path.append(submodule)

from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray
from dynamixel_sdk import *                    # Uses Dynamixel SDK library
from Dynamixel import *                        # Dynamixel motor class                                  
from dyn_functions import *                    # Dynamixel support functions
from turtle_controller import *                # Controller 
from Constants import *                        # File of constant variables
from Mod import *
from utilities import *
import math
from math import cos, sin
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt
from utilities import *
import json
import traceback
from queue import Queue
import serial

def save_data(acc_data, gyr_data, mag_data, timestamps):
    t = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    folder_name =  "sensor_data/" + t
    os.makedirs(folder_name, exist_ok=True)
    scipy.io.savemat(folder_name + "/data.mat", {'acc_data': acc_data.T,'gyr_data': gyr_data.T,'mag_data': mag_data.T, 'time_data': timestamps})


def main(args=None):
    """
    Saving everything in a MATLAB file for ros testing with ppo node before real turtle testing is done
    """
    # {"Acc":[  89.36,  29.30,  1024.90 ], "Gyr" :[  0.52, -0.55,  0.07 ], "Mag" :[ -54.45, -10.65,  48.30 ],"Voltage": [ 0.04 ]}
    n_axis = 3
    acc_data = np.zeros((n_axis, 1))
    gyr_data = np.zeros((n_axis,1))
    mag_data = np.zeros((n_axis,1))
    timestamps = np.zeros((1,1))
    delta_t = 15
    t_0 = time.time()
    time_elapsed = 0
    while time_elapsed < delta_t:
        xiao =  serial.Serial('/dev/ttyACM0', 115200, timeout=3)   
        sensors = xiao.readline()
        sensor_dict = json.loads(sensors.decode('utf-8'))

        # add time stamp
        t = time.time()
        time_elapsed = t-t_0
        timestamps = np.append(timestamps, time_elapsed) 

        # add sensor data
        acc_data = np.append(acc_data, sensor_dict['Acc'])
        gyr_data = np.append(gyr_data, sensor_dict['Gyr'])
        mag_data = np.append(mag_data, sensor_dict['Mag'])
    print("recorded data to folder")
    save_data(acc_data=acc_data, gyr_data=gyr_data, mag_data=mag_data, timestamps=timestamps)
if __name__=='__main__':
    main()