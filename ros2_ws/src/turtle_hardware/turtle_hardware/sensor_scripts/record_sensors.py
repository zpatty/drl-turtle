import rclpy
from rclpy.node import Node
from rclpy.executors import Executor, MultiThreadedExecutor
from rcl_interfaces.msg import SetParametersResult
from turtle_interfaces.msg import TurtleTraj
import json
import time
import scipy
import sys
import os

submodule = os.path.expanduser("~") + "/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/sensor_scripts"
sys.path.append(submodule)

from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray
import math
from math import cos, sin
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt
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
    voltage_data = np.zeros((1,1))
    quat_data = np.zeros((4,1))
    delta_t = 15
    t_0 = time.time()
    time_elapsed = 0
    xiao =  serial.Serial('/dev/ttyACM0', 115200, timeout=3)   
    xiao.reset_input_buffer()

    while time_elapsed < delta_t:
        # if xiao.in_waiting > 0:
        #     line = xiao.readline().decode('utf-8').rstrip()
        #     print(line)
        # no_check = False
        sensors = xiao.readline()
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
                        acc_data = np.append(acc_data, acc, axis=1)
                        gyr_data = np.append(gyr_data, gyr, axis=1)
                        quat_data = np.append(quat_data, quat, axis=1)
                        voltage_data = np.append(voltage_data, volt)
                        voltage = volt
    # print("recorded data to folder")
    # save_data(acc_data=acc_data, gyr_data=gyr_data, mag_data=mag_data, timestamps=timestamps)
if __name__=='__main__':
    main()