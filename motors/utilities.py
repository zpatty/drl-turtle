#!/usr/bin/env python
# -*- coding: utf-8 -*-
from datetime import datetime
import os
import numpy as np
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

def parse_setpoint(nq):
    with open('q.json') as q_json:
        param = json.load(q_json)
    qd = np.zeros((nq,1))
    for i in range(nq):
        qd[i] = ['q' + str(i)]
        
    qd_params = json.dumps(param, indent=14)
    
    return qd
