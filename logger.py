#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import time
from datetime import datetime
from math import cos, sin
import numpy as np
import scipy.io
import pandas as pd
from dynamixel_sdk import *                    # Uses Dynamixel SDK library
from Dynamixel import *                        # Dynamixel motor class


def log_config(config, head):
    """
    Input: current config, including timestamp
    """
    # columns in csv 
    mod_cols = ['Time', 'dx', 'dy', 'dL']
    data = {
        'Time': [config[0]],
        'dx': [config[1]],
        'dy': [config[2]],
        'dL': [config[3]]
    }
    mod_config = pd.DataFrame(data)
    mod_config.to_csv('mod_config.csv', mode='a', index=False, header=head)
    mod_config.columns = mod_cols

def log_torques(torques, head):
    """
    Function that writes the torques the controller sends to the module's motors
    into a csv file.
    """
    # columns in csv 
    cols = ['Time', 'M1', 'M2', 'M3']
    data = {
        'Time': [torques[0]],
        'M1': [torques[1]],
        'M2': [torques[2]],
        'M3': [torques[3]]
    }
    mod_torques = pd.DataFrame(data)
    mod_torques.to_csv('mod_torques.csv', mode='a', index=False, header=head)
    mod_torques.columns = cols
# def log_config_esolo(config, head):
#     """
#     Input: current config, including timestamp
#     """
#     # columns in csv 
#     mod_cols = ['Time', 'th1', 'th2', 'th3', 'th4', 'th5', 'th6', 'th7']
#     data = {
#         'Time': [config[0]],
#         'dx': [config[1]],
#         'dy': [config[2]],
#         'dL': [config[3]]
#     }
#     mod_config = pd.DataFrame(data)
#     mod_config.to_csv('mod_config.csv', mode='a', index=False, header=head)
#     mod_config.columns = mod_cols

def log_qd(qds, head):
    """
    Function that writes the current desired q into a csv file.
    """
    # columns in csv 
    cols = ['Time', 'dx', 'dy', 'dL']
    data = {
        'Time': [qds[0]],
        'dx': [qds[1]],
        'dy': [qds[2]],
        'dL': [qds[3]]
    }
    mod_qds = pd.DataFrame(data)
    mod_qds.to_csv('mod_qd.csv', mode='a', index=False, header=head)
    mod_qds.columns = cols

def log_lens(l, head):
    """
    Function that writes the current cable lengths into a csv file.
    """
    # columns in csv 
    cols = ['Time', 'l1', 'l2', 'l3']
    data = {
        'Time': [l[0]],
        'l1': [l[1]],
        'l2': [l[2]],
        'l3': [l[3]]
    }
    mod_lens = pd.DataFrame(data)
    mod_lens.to_csv('mod_len.csv', mode='a', index=False, header=head)
    mod_lens.columns = cols

def log_motor_angles(th):
    """
    Function that writes the current motor angles to a csv file. We can use this csv file
    to recover the motor positions in the case that the encoders reset at startup.
    TODO: we may only need the dthetas?
    We know that at startup, our th0s change, but our cable lengths do not.
    """
    # columns in csv 
    cols = ['th1', 'th2', 'th3']
    data = {
        'th1': [th[0]],
        'th2': [th[1]],
        'th3': [th[2]]
    }
    mod_ths = pd.DataFrame(data)
    mod_ths.to_csv('mod_angles.csv', mode='a', index=False, header=head)
    mod_ths.columns = cols