import os
import sys
import time
# import torch
import rclpy
import numpy as np
from EPHE import EPHE
from AukeCPG import AukeCPG
from TurtleRobot import *
import random

# from CPG_gym import *

from std_msgs.msg import String
from turtle_dynamixel.utilities import *

os.system('sudo /home/crush/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/turtle_dynamixel/latency_write.sh')

global mode
mode = 'rest'

def main(args=None):
    rclpy.init(args=args)
    threshold = 11.3
    params, config_params = parse_learning_params()
    turtle_node = TurtleRobot('turtle_mode_cmd', params=params)
    q = np.array(turtle_node.Joints.get_position()).reshape(-1,1)
    print(f"Our initial q: " + str(q))
    # create folders 
    try:
        typed_name =  input("give folder a name: ")
        folder_name = "data/" + typed_name
        os.makedirs(folder_name, exist_ok=False)

    except:
        print("received weird input or folder already exists-- naming folder with timestamp")
        t = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        folder_name =  "data/" + t
        os.makedirs(folder_name)
    create_shell_script(params=params, folder_name=folder_name)

    best_param_fname = folder_name + f'/turtle_node.best_params.pth'
    config_fname = folder_name + "/config.json"
    # create shell script for sending data over
    
    num_params = params["num_params"]
    num_mods = params["num_mods"]
    a_r  = params["a_r"]
    a_x = params["a_x"]
    phi = params["phi"]
    w = params["w"]
    M = params["M"]
    K = params["K"]
    dt = params["dt"]
    mu = np.array(params["mu"])
    sigma = np.array(params["sigma"])
    max_episodes = params["max_episodes"]
    max_episode_length = params["max_episode_length"]
    num_priors = params["num_priors"]
    turtle_node.best_params = np.zeros((num_params))

    config_params = json.dumps(params, indent=10)

    # save config file to specified folder beforehand
    with open(config_fname, 'w') as outfile:
        outfile.write(config_params)

    # data structs
    param_data = np.zeros((num_params, M, max_episodes))
    mu_data = np.zeros((num_params, max_episodes))
    sigma_data = np.zeros((num_params, max_episodes))
    reward_data = np.zeros((max_episode_length, M, max_episodes))
    cpg_data = np.zeros((10, max_episode_length, M, max_episodes))
    time_data = np.zeros((max_episode_length, M, max_episodes))
    q_data = np.zeros((10, max_episode_length, M, max_episodes))
    dq_data = np.zeros((10, max_episode_length, M, max_episodes))
    tau_data = np.zeros((10, max_episode_length, M, max_episodes))
    voltage_data = np.zeros((max_episode_length, M, max_episodes))
    acc_data = np.zeros((3, max_episode_length, M, max_episodes))
    gyr_data = np.zeros((3, max_episode_length, M, max_episodes))
    quat_data = np.zeros((4, max_episode_length, M, max_episodes))

    best_reward = 0
    try: 
        while rclpy.ok():
            rclpy.spin_once(turtle_node)
            print(f"turtle mode: {turtle_node.mode}\n")
            if turtle_node.voltage < threshold:
                print("[WARNING] volt reading too low--closing program....")
                turtle_node.Joints.disable_torque()
                break
            if turtle_node.mode == 'stop':
                print("ending entire program...")
                print("disabling torques entirely...")
                turtle_node.Joints.send_torque_cmd([0] *len(turtle_node.IDs))
                turtle_node.Joints.disable_torque()
                turtle_node.xiao.close()

                cmd_msg = String()
                cmd_msg.data = "stop_received"
                turtle_node.cmd_received_pub.publish(cmd_msg)
                print("sent stop received msg")
                break
            elif turtle_node.mode == 'rest':
                rclpy.spin_once(turtle_node)
                turtle_node.read_voltage()
                turtle_node.read_sensors()
                print(f"quat data: {turtle_node.quat_data[:, -1]}")
                if turtle_node.voltage < threshold:
                    turtle_node.Joints.disable_torque()
                    print(f"VOLTAGE: {turtle_node.voltage}")
                    print("THRESHOLD MET TURN OFFFF")
                    break
                turtle_node.Joints.disable_torque()
                print(turtle_node.Joints.get_position())
                cmd_msg = String()
                cmd_msg.data = 'rest_received'
                turtle_node.cmd_received_pub.publish(cmd_msg)
                print(f"current voltage: {turtle_node.voltage}\n")
            elif turtle_node.mode == 'Auke':
                # Auke CPG model that outputs position in radians
                print("Auke CPG training time\n")
                cpg = AukeCPG(num_params=num_params, num_mods=num_mods, phi=phi, w=w, a_r=a_r, a_x=a_x, dt=dt)
                # data structs for plotting
                prior_acc_data = np.zeros((3, max_episode_length, num_priors))
                prior_gyr_data = np.zeros((3, max_episode_length, num_priors))
                prior_quat_data = np.zeros((4, max_episode_length, num_priors))
                prior_voltage_data = np.zeros((max_episode_length, num_priors))
                prior_reward_data = np.zeros((max_episode_length, num_priors))
                prior_q_data = np.zeros((10, max_episode_length, num_priors))
                prior_dq_data = np.zeros((10, max_episode_length, num_priors))
                prior_tau_data = np.zeros((10, max_episode_length, num_priors))
                prior_cpg_data = np.zeros((10, max_episode_length, num_priors))
                prior_time_data = np.zeros((max_episode_length, num_priors))
                best_reward = 0
                # first run priors sweep ----------------------------------------------------------------------------------------------------------
                prior_param_data = np.zeros((num_params, num_priors))
                R = np.zeros(num_priors)
                for i in range(num_priors):
                    rclpy.spin_once(turtle_node)
                    if turtle_node.mode == 'rest' or turtle_node.mode == 'stop' or turtle_node.voltage < threshold:
                        print("breaking out of rollout----------------------------------------------------------------------")
                        turtle_node.Joints.send_torque_cmd([0] *len(turtle_node.IDs))
                        turtle_node.Joints.disable_torque()
                        break
                    cpg.reset()

                    freq = np.random.uniform(low=2.0, high=3.5, size=1)
                    shoulder = np.random.uniform(low=0.4, high=0.7, size=1)
                    mid = np.random.uniform(low=0.7, high=0.9, size=1)
                    flipper = np.random.uniform(low=0.3, high=0.6   , size=1)

                    b_joint = np.random.uniform(low=0.0, high=0.1, size=1)
                    b_fin = np.random.uniform(low=0.0, high=0.1, size=1)
                    # amps = np.random.uniform(low=0.1, high=1, size=10)
                    offset_shoulder = np.random.uniform(low=3.0, high=3.05, size=1)
                    offset_mid = np.random.uniform(low=3.14, high=3.18, size=1)
                    offset_flipper = np.random.uniform(low=3.14, high=3.15, size=1)

                    offset_bjoint = np.random.uniform(low=3.14, high=3.15, size=1)
                    offset_bfin = np.random.uniform(low=3.14, high=3.15, size=1)

                    sol = np.concatenate((freq, shoulder, mid, flipper, shoulder, mid, flipper, b_joint, b_fin, b_joint, b_fin, 
                                            offset_shoulder, offset_mid, offset_flipper, offset_shoulder, offset_mid, offset_flipper,
                                            offset_bjoint, offset_bfin, offset_bjoint, offset_bfin))                    
                    prior_param_data[:, i] = sol
                    cpg.set_parameters(sol)
                    observation, __ = turtle_node.reset()
                    cumulative_reward = 0.0
                    t = 0
                    timestamps = np.zeros(max_episode_length)
                    t_0 = time.time()
                    while True:
                        rclpy.spin_once(turtle_node)
                        if turtle_node.mode == 'rest' or turtle_node.mode == 'stop' or turtle_node.voltage < threshold:
                            print("breaking out of rollout----------------------------------------------------------------------")
                            turtle_node.Joints.send_torque_cmd([0] *len(turtle_node.IDs))
                            turtle_node.Joints.disable_torque()
                            break
                        action = cpg.get_action()
                        prior_cpg_data[:, t, i] = action
                        # for a in range(len(action)):
                        #     if a not in [6,7,8,9]:
                        #         action[a] = action[a] + turtle_node.center_pos[a]
                        timestamps[t] = time.time()-t_0 
                        observation, reward, terminated, truncated, info = turtle_node.step(action, PD=True)
                        v, tau = info
                        prior_tau_data[:, t, i] = tau
                        prior_q_data[:, t, i] = observation
                        prior_dq_data[:, t, i] = v
                        prior_reward_data[t, i] = reward
                        cumulative_reward += reward
                        t += 1
                        if t >= max_episode_length:
                            turtle_node.Joints.disable_torque()
                            print(f"---------------prior {i}, reward: {cumulative_reward}\n")
                            print(f"quat weight: {turtle_node.tot_quat}")
                            print(f"acc weight: {turtle_node.tot_acc}")
                            print(f"tau weight: {turtle_node.tot_tau}")

                            break
                    try:
                        prior_time_data[:, i] = timestamps
                        prior_acc_data[:, :, i] = turtle_node.acc_data[:, 1:]
                        prior_gyr_data[:, :, i] = turtle_node.gyr_data[:, 1:]
                        prior_quat_data[:, :, i] = turtle_node.quat_data[:, 1:]
                        prior_voltage_data[:, i] = turtle_node.voltage_data[1:]
                    except:
                        print(f"stopped mid rollout-- saving everything but this rollout data")

                    # save to folder after each rollout
                    scipy.io.savemat(folder_name + "/prior_data.mat", {
                        'param_data': prior_param_data, 'reward_data': prior_reward_data, 'q_data': prior_q_data, 'dq_data': prior_dq_data,
                        'tau_data': prior_tau_data, 'voltage_data': prior_voltage_data, 'acc_data': prior_acc_data, 'quat_data': prior_quat_data, 
                        'gyr_data': prior_gyr_data, 'time_data': prior_time_data, 'cpg_data': prior_cpg_data})
                    R[i] = cumulative_reward
                # get indices of K best rewards
                best_inds = np.argsort(-R)[:K]
                k_params = prior_param_data[:, best_inds]
                print(f"k params: {k_params}")
                k_rewards = R[best_inds]
                print(f"k rewards: {k_rewards}")
                mu = k_params[:, 0]
                # mu = np.array(params["mu"])
                sigma = np.ones(num_params) * 0.15
                
                print(f"using this mu: {mu}\n")
                prior_mu = mu
                ## RESET CPG ##
                cpg.reset()
################# reinitialize EPHE ################################
                ephe = EPHE(
                    
                    # We are looking for solutions whose lengths are equal
                    # to the number of parameters required by the policy:
                    solution_length=mu.shape[0],
                    
                    # Population size: the number of trajectories we run with given mu and sigma 
                    popsize=M,
                    
                    # Initial mean of the search distribution:
                    center_init=mu,
                    
                    # Initial standard deviation of the search distribution:
                    stdev_init=sigma,

                    # dtype is expected as float32 when using the policy objects
                    dtype='float32', 

                    K=K
                )

                # get to training
                turtle_node.Joints.enable_torque()
    ############################ EPISODE #################################################
                for episode in range(max_episodes):
                    # data folder for this episode
                    rclpy.spin_once(turtle_node)
                    if turtle_node.mode == 'rest' or turtle_node.mode == 'stop' or turtle_node.voltage < threshold:
                        turtle_node.Joints.send_torque_cmd([0] *len(turtle_node.IDs))
                        turtle_node.Joints.disable_torque()
                        print("saving data to turtle...")
                        # save data structs to matlab 
                        scipy.io.savemat(folder_name + "/data.mat", {'prior_mu': prior_mu,'mu_data': mu_data,'sigma_data': sigma_data,
                                            'param_data': param_data, 'reward_data': reward_data, 'q_data': q_data, 'dq_data': dq_data,
                                            'tau_data': tau_data, 'voltage_data': voltage_data, 'acc_data': acc_data, 'quat_data': quat_data, 
                                            'gyr_data': gyr_data, 'time_data': time_data, 'cpg_data': cpg_data})
                        scp_data()
                        break
                    print(f"--------------------episode: {episode} of {max_episodes}------------------")
                    lst_params = np.zeros((num_params, M))
                    solutions = ephe.ask()     
                    param_data[:, :, episode] = solutions.T
                    # rewards from M rollouts   
                    R = np.zeros(M)
    ###################### run your M rollouts########################################################################3
                    for m in range(M):
                        print(f"--------------------episode {episode} of {max_episodes}, rollout: {m} of {M}--------------------")
                        rclpy.spin_once(turtle_node)
                        if turtle_node.mode == 'rest' or turtle_node.mode == 'stop' or turtle_node.voltage < threshold:
                            print("breaking out of episode----------------------------------------------------------------------")
                            turtle_node.Joints.send_torque_cmd([0] *len(turtle_node.IDs))
                            turtle_node.Joints.disable_torque()
                            break
                        lst_params[:, m] = cpg.filter(solutions[m])
                        cpg.set_parameters(solutions[m])
                        cumulative_reward = 0.0

                        # reset the environment after every rollout
                        timestamps = np.zeros(max_episode_length)
                        t = 0                       # to ensure we only go max_episode length
                        observation, __ = turtle_node.reset()
                        t_0 = time.time()
    ############################ ROLL OUT ###############################################################################
                        while True:
                            rclpy.spin_once(turtle_node)
                            if turtle_node.mode == 'rest' or turtle_node.mode == 'stop' or turtle_node.voltage < threshold:
                                print("breaking out of rollout----------------------------------------------------------------------")
                                turtle_node.Joints.send_torque_cmd([0] *len(turtle_node.IDs))
                                turtle_node.Joints.disable_torque()
                                break
                            # action = cpg_actions[:, t]
                            action = cpg.get_action()
                            # for a in range(len(action)):
                            #     action[a] = action[a] + turtle_node.center_pos[a]
                            cpg_data[:, t, m, episode] = action
                            # save the raw cpg output
                            timestamps[t] = time.time()-t_0 
                            observation, reward, terminated, truncated, info = turtle_node.step(action, PD=True)
                            v, tau = info
                            done = truncated or terminated
                            reward_data[t, m, episode] = reward
                            tau_data[:, t, m, episode] = tau
                            q_data[:, t, m, episode] = observation
                            dq_data[:, t, m, episode] = v
                            cumulative_reward += reward
                            t += 1
                            if t >= max_episode_length:
                                turtle_node.Joints.disable_torque()
                                print("\n\n")
                                print(f"---------------rollout reward: {cumulative_reward}\n\n\n\n")
                                print(f"quat weight: {turtle_node.tot_quat}")
                                print(f"acc weight: {turtle_node.tot_acc}")
                                print(f"tau weight: {turtle_node.tot_tau}")
                                break
                        try:
                        # record data from rollout
                            time_data[:, m, episode] = timestamps
                            acc_data[:, :, m, episode] = turtle_node.acc_data[:, 1:]
                            gyr_data[:, :, m, episode] = turtle_node.gyr_data[:, 1:]
                            quat_data[:, :, m, episode] = turtle_node.quat_data[:, 1:]
                            voltage_data[:, m, episode] = turtle_node.voltage_data[1:]
                        except:
                            print(f"stopped mid rollout-- saving everything but this rollout data")
                        # save to folder after each rollout
                        scipy.io.savemat(folder_name + "/data.mat", {'prior_mu': prior_mu, 'mu_data': mu_data,'sigma_data': sigma_data,
                            'param_data': param_data, 'reward_data': reward_data, 'q_data': q_data, 'dq_data': dq_data,
                            'tau_data': tau_data, 'voltage_data': voltage_data, 'acc_data': acc_data, 'quat_data': quat_data, 
                            'gyr_data': gyr_data, 'time_data': time_data, 'cpg_data': cpg_data})

                        # replaces set_params_and_run function for now
                        if cumulative_reward < 0:
                            fitness = 0
                        else:
                            fitness = cumulative_reward
                        if fitness > best_reward:
                            print(f"params with best fitness: {solutions[m]} with reward {fitness}\n")
                            best_reward = fitness
                            turtle_node.best_params = solutions[m]  
                        # update reward array for updating mu and sigma
                        R[m] = fitness
                    print("--------------------- Episode:", episode, "  median score:", np.median(R), "------------------")
                    print(f"all rewards: {R}\n")
                    # get indices of K best rewards
                    best_inds = np.argsort(-R)[:K]
                    k_params = lst_params[:, best_inds]
                    print(f"k params: {k_params}")
                    k_rewards = R[best_inds]
                    print(f"k rewards: {k_rewards}")
                    # We inform our ephe solver of the fitnesses we received, so that the population gets updated accordingly.
                    # BUT we only update if all k rewards are positive
                    update = False
                    for g in k_rewards:
                        if g > 0:
                            update = True
                    if update:
                        ephe.update(k_rewards=k_rewards, k_params=k_params)
                        print(f"---------------------new mu: {ephe.center()}---------------------------\n")
                        print(f"--------------------new sigma: {ephe.sigma()}--------------------------\n")
                    # save mus and sigmas
                    mu_data[:, episode] = ephe.center()
                    sigma_data[:, episode] = ephe.sigma()
                    best_mu = ephe.center()
                    best_sigma = ephe.sigma()
                turtle_node.Joints.disable_torque()
                print(f"best mu: {best_mu}\n")
                print(f"best sigma: {best_sigma}\n")
                print(f"best params: {turtle_node.best_params} got reward of {best_reward}\n")
                print("------------------------Saving data------------------------\n")
                print("saving data....")
                # save data structs to matlab for this episode
                scipy.io.savemat(folder_name + "/data.mat", {'mu_data': mu_data,'sigma_data': sigma_data,
                                            'param_data': param_data, 'reward_data': reward_data, 'q_data': q_data, 'dq_data': dq_data,
                                            'tau_data': tau_data, 'voltage_data': voltage_data, 'acc_data': acc_data, 'quat_data': quat_data, 
                                            'gyr_data': gyr_data, 'time_data': time_data, 'cpg_data': cpg_data})
                break

            elif turtle_node.mode == 'planner':
                """
                Randomly pick a motion primitive and run it 4-5 times
                """
                # primitives = ['surface', 'turnrr', 'straight', 'turnlr']
                primitives = ['straight']
                num_cycles = 4
                turtle_node.Joints.disable_torque()
                turtle_node.Joints.set_current_cntrl_mode()
                turtle_node.Joints.enable_torque()

                while True:
                    rclpy.spin_once(turtle_node)
                    if turtle_node.mode == 'rest' or turtle_node.mode == 'stop' or turtle_node.voltage < threshold:
                        turtle_node.Joints.send_torque_cmd([0] *len(turtle_node.IDs))
                        turtle_node.Joints.disable_torque()
                        break
                    primitive = random.choice(primitives)
                    print(f"---------------------------------------PRIMITIVE: {primitive}\n\n")
                    qd_mat = mat2np(f'/home/crush/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/turtle_trajectory/{primitive}/qd.mat', 'qd')
                    dqd_mat = mat2np(f'/home/crush/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/turtle_trajectory/{primitive}/dqd.mat', 'dqd')
                    ddqd_mat = mat2np(f'/home/crush/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/turtle_trajectory/{primitive}/ddqd.mat', 'ddqd')
                    tvec = mat2np(f'/home/crush/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/turtle_trajectory/{primitive}/tvec.mat', 'tvec')
                    first_time = True
                    first_loop = True
                    input_history = np.zeros((turtle_node.nq,10))
                    q_data = np.zeros((turtle_node.nq,1))
                    tau_data = np.zeros((turtle_node.nq,1))
                    timestamps = np.zeros((1,1))
                    dt_loop = np.zeros((1,1))       # hold dt data 
                    dq_data = np.zeros((turtle_node.nq,1))
                    tau_data = np.zeros((turtle_node.nq,1))
                    timestamps = np.zeros((1,1))
                    dt_loop = np.zeros((1,1))       # hold dt data 
                    cycle = 0
                    t_record = time.time()
                    # zero =  np.zeros((self.nq,1))
                    t_old = time.time()
                    # our loop's "starting" time
                    t_0 = time.time()
                    while cycle < num_cycles:
                        if turtle_node.voltage < threshold:
                            print("voltage too low--powering off...")
                            turtle_node.Joints.disable_torque()
                            break
                        rclpy.spin_once(turtle_node)
                        timestamps = np.append(timestamps, time.time()-t_record)
                        # print("traj 1...")
                        if turtle_node.mode == 'rest' or turtle_node.mode == 'stop':
                            turtle_node.Joints.send_torque_cmd(turtle_node.nq * [0])
                            turtle_node.Joints.disable_torque()
                            # try:
                            # # record data from rollout
                            #     time_data= timestamps
                            #     acc_data= turtle_node.acc_data[:, 1:]
                            #     gyr_data= turtle_node.gyr_data[:, 1:]
                            #     quat_data= turtle_node.quat_data[:, 1:]
                            #     voltage_data = turtle_node.voltage_data[1:]
                            # except:
                            #     print(f"stopped mid rollout-- saving everything but this rollout data")
                            mu_data =np.zeros((1,1))   
                            reward_data = np.zeros((1,1))   
                            param_data = np.zeros((1,1))   
                            sigma_data = np.zeros((1,1))   
                            cpg_data = np.zeros((1,1))   
                            # save to folder after each rollout
                            scipy.io.savemat(folder_name + "/data.mat", {'mu_data': mu_data,'sigma_data': sigma_data,
                                'param_data': param_data, 'reward_data': reward_data, 'q_data': q_data, 'dq_data': dq_data,
                                'tau_data': tau_data, 'voltage_data': turtle_node.voltage_data, 'acc_data': turtle_node.acc_data, 'quat_data': turtle_node.quat_data, 
                                'gyr_data': turtle_node.gyr_data, 'time_data': timestamps, 'cpg_data': cpg_data})                            
                            scp_data()
                            first_time = True
                            break
                        turtle_node.read_sensors()

                        q = np.array(turtle_node.Joints.get_position()).reshape(-1,1)
                        if first_loop:
                            n = get_qindex((time.time() - t_0), tvec)
                        else:
                            # print("done with first loop")
                            offset = t_0 - 2
                            n = get_qindex((time.time() - offset), tvec)

                        # print(f"n: {n}\n")
                        if n == len(tvec[0]) - 1:
                            first_loop = False
                            t_0 = time.time()
                            cycle += 1
                            print(f"-----------------cycle: {cycle}\n\n\n")
                        
                        qd = np.array(qd_mat[:, n]).reshape(-1,1)
                        dqd = np.array(dqd_mat[:, n]).reshape(-1,1)
                        ddqd = np.array(ddqd_mat[:, n]).reshape(-1,1)
                        # # print(f"[DEBUG] qdata: {q_data}\n")
                        # print(f"[DEBUG] qd: {qd}\n")
                        q_data=np.append(q_data, q, axis = 1) 
                        # # At the first iteration velocity is 0  
                        
                        if first_time:
                            # dq = np.zeros((nq,1))
                            dq = np.array(turtle_node.Joints.get_velocity()).reshape(-1,1)
                            dq_data=np.append(dq_data, dq, axis = 1) 
                            q_old = q
                            first_time = False
                        else:
                            t = time.time()
                            dt = t - t_old
                        #     # print(f"[DEBUG] dt: {dt}\n")  
                            t_old = t
                            # dq = diff(q, q_old, dt)
                            dq = np.array(turtle_node.Joints.get_velocity()).reshape(-1,1)
                            dq_data=np.append(dq_data, dq, axis = 1) 
                            q_old = q
                            # # calculate errors
                        err = q - qd
                        # # print(f"[DEBUG] e: {err}\n")
                        # # print(f"[DEBUG] q: {q * 180/3.14}\n")
                        # # print(f"[DEBUG] qd: {qd * 180/3.14}\n")
                        err_dot = dq

                        tau = turtle_controller(q,dq,qd,dqd,ddqd,turtle_node.Kp,turtle_node.KD)

                        # publish motor data, tau data, 
                        
                        input_history = np.append(input_history[:,1:], tau,axis=1)
                        input_mean = np.mean(input_history, axis = 1)
                        curr = grab_arm_current(input_mean, min_torque, max_torque)
                        turtle_node.Joints.send_torque_cmd(curr)

                turtle_node.Joints.disable_torque()

            else:
                print("wrong command received....")
    except Exception as e:
        print("some error occurred")
        turtle_node.Joints.send_torque_cmd([0] * len(turtle_node.IDs))
        turtle_node.Joints.disable_torque()
        turtle_node.xiao.close()
        exec_type, obj, tb = sys.exc_info()
        fname = os.path.split(tb.tb_frame.f_code.co_filename)[1]
        print(exec_type, fname, tb.tb_lineno)
        print(e)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

