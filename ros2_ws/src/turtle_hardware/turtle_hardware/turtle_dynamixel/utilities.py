#!/usr/bin/env python
# -*- coding: utf-8 -*-
from datetime import datetime
import os
import sys
import time
import json
import scipy
import subprocess
import numpy as np

if os.name == 'nt':
    import msvcrt
    def getch():
        return msvcrt.getch().decode()
    def kbhit():
        return msvcrt.kbhit()
else:
    import termios
    from select import select
    fd = sys.stdin.fileno()
    old_term = termios.tcgetattr(fd)
    new_term = termios.tcgetattr(fd)

    def getch():
        new_term[3] = (new_term[3] & ~termios.ICANON & ~termios.ECHO)
        termios.tcsetattr(fd, termios.TCSANOW, new_term)
        try:
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_term)
        return ch

    def kbhit():
        new_term[3] = (new_term[3] & ~(termios.ICANON | termios.ECHO))
        termios.tcsetattr(fd, termios.TCSANOW, new_term)
        try:
            dr,dw,de = select([sys.stdin], [], [], 0)
            if dr != []:
                return 1
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_term)
            sys.stdout.flush()

        return 0

def save_data(q_data, qd, tau_data, timestamps, config_params, dt_loop):

    print(f"time since: {time.time() - t_0}\n")
    t = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    folder_name =  t
    os.makedirs(folder_name, exist_ok=True)
    q_data_name = folder_name + "/qs.mat"
    tau_data_name = folder_name + "/taus.mat"
    time_data_name = folder_name + "/timestamps.mat"
    qd_name = folder_name + "/q_desired.mat"

    scipy.io.savemat(q_data_name, {'q_data': q_data.T})
    scipy.io.savemat(tau_data_name, {'tau_data': tau_data.T})
    scipy.io.savemat(time_data_name, {'time_data': timestamps})
    scipy.io.savemat(qd_name, {'q_desired': qd})
    new_config = folder_name + "/config.json"
    with open(new_config, "w") as outfile:
        outfile.write(config_params)
                # Writing to new config.json
    print(f"[OUTPUT] Our desired config: {qd}\n")
    print(f"[OUTPUT] Our last recorded q: {q_data[:,-1]}\n")
    print(f"max dt value: {np.max(dt_loop)}\n")
    print(f"last time: {timestamps[-1]}\n")

def parse_config():
    with open('config.json') as config:
        param = json.load(config)
    print(f"[MESSAGE] Config: {param}\n")
    Kp = ['Kp']
    KD = ['KD']
    
    # Serializing json
    config_params = json.dumps(param, indent=14)
    
    return Kp, KD, config_params

def parse_learning_params():
    with open('turtle_config/config.json') as config:
        param = json.load(config)
    print(f"[MESSAGE] Config: {param}\n")    
    # Serializing json
    config_params = json.dumps(param, indent=14)
    
    return param, config_params


def parse_setpoint(nq):
    with open('q.json') as q_json:
        param = json.load(q_json)
    qd = np.zeros((nq,1))
    for i in range(nq):
        qd[i] = ['q' + str(i)]
        
    qd_params = json.dumps(param, indent=14)
    
    return qd

def create_shell_script(params, folder_name):
    """
    Creates temporary shell script that sends data from turtle machine to remote machine
    """
    local_folder_path = "/home/crush/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/" + folder_name
    remote_user = params["remote_user"]
    remote_host = params["remote_host"]
    remote_folder_path = params["remote_folder_path"] + folder_name

    shell_script_content = f"""#!/bin/bash
    scp -r {local_folder_path} {remote_user}@{remote_host}:{remote_folder_path}
    """

    # Write the shell script to a temporary file
    with open('send_data.sh', 'w') as shell_script:
        shell_script.write(shell_script_content)

    # Make the shell script executable
    subprocess.run(['chmod', '+x', 'send_data.sh'])

def scp_data():
    """
    Now that data was saved locally, run shell script
    """
    # Execute the shell script
    subprocess.run(['./send_data.sh'])

    # Clean up the temporary shell script file
    subprocess.run(['rm', 'send_data.sh'])



def create_shell_script(params, folder_name):
    """
    Creates temporary shell script that sends data from turtle machine to remote machine
    """
    local_folder_path = "/home/crush/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/" + folder_name
    remote_user = params["remote_user"]
    remote_host = params["remote_host"]
    remote_folder_path = params["remote_folder_path"] + folder_name

    shell_script_content = f"""#!/bin/bash
    scp -r {local_folder_path} {remote_user}@{remote_host}:{remote_folder_path}
    """

    # Write the shell script to a temporary file
    with open('send_data.sh', 'w') as shell_script:
        shell_script.write(shell_script_content)

    # Make the shell script executable
    subprocess.run(['chmod', '+x', 'send_data.sh'])

def scp_data():
    """
    Now that data was saved locally, run shell script
    """
    # Execute the shell script
    subprocess.run(['./send_data.sh'])

    # Clean up the temporary shell script file
    subprocess.run(['rm', 'send_data.sh'])

