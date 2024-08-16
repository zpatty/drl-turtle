from __future__ import print_function
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from turtle_dynamixel.dyn_functions import *                 

submodule = os.path.expanduser("~") + "/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/"
sys.path.append(submodule)

def main(args=None):
    """
    q shape:      (num_mods, rollout_len, num rollouts (M), num_episodes)
    dq shape:     (num_mods, rollout_len, num rollouts (M), num_episodes)
    tau shape:    (num_mods, rollout_len, num rollouts (M), num_episodes)
    cpg shape:    (num_mods, rollout_len, num rollouts (M), num_episodes)
    reward shape: (rollout_len, num rollouts (M), num_episodes)
    acc shape:    (3 axis, rollout_len, num rollouts (M), num_episode)
    quat shape:   (4 axis, rollout_len, num rollouts (M), num_episode)
    gyr shape:    (3 axis, rollout_len, num rollouts (M), num_episode)
    """

    FNAME = 'turtle_data/Auke_pooltest_3'
    q_data = mat2np(FNAME + '/data.mat', 'q_data')
    dq_data = mat2np(FNAME + '/data.mat', 'dq_data')
    tau_data = mat2np(FNAME + '/data.mat', 'tau_data')
    cpg_data = mat2np(FNAME + '/data.mat', 'cpg_data')
    reward_data = mat2np(FNAME + '/data.mat', 'reward_data')
    acc_data = mat2np(FNAME + '/data.mat', 'acc_data')
    quat_data = mat2np(FNAME + '/data.mat', 'quat_data')
    gyr_data = mat2np(FNAME + '/data.mat', 'gyr_data')
    tau_data = mat2np(FNAME + '/data.mat', 'tau_data')
    param_data = mat2np(FNAME + '/data.mat', 'param_data')

    print(f"q shape: {q_data.shape}")
    print(f"dq shape: {dq_data.shape}")
    print(f"tau shape: {tau_data.shape}")
    print(f"cpg shape: {cpg_data.shape}")
    print(f"reward shape: {reward_data.shape}")
    print(f"acc shape: {acc_data.shape}")
    print(f"quat shape: {quat_data.shape}")
    print(f"gyr shape: {gyr_data.shape}")
    print(f"tau shape: {tau_data.shape}")
    print(f"param shape: {param_data.shape}")
    rollout_rewards = np.sum(reward_data, axis=0)
    print(f"rollout reward dim: {rollout_rewards.shape}\n")
    ep = 4
    print(f"rollout rewards: {rollout_rewards[:, ep]}\n")
    print(f"params: {param_data[:, :, ep]}")
    max_index = np.argmax(reward_data)
    max_indices = np.unravel_index(reward_data.argmax(), reward_data.shape)
    print(f"max indices: {max_indices}\n")
    # print(f"test: {reward_data[76, 36, 1]}\n")

    # print(f"Acc data: {acc_data[2, :, 1, 1]}")
    rollout_len = q_data.shape[1]
    num_rollouts = q_data.shape[2]
    dt = 0.01
    rollout = 0
    episode = 0
    num_rollouts = 1        # number of rollouts we wanna look at
    # plot reward data 
    # t = np.arange(0, num_rollouts, 1)
    # print(f"t shape: {t.shape}")
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 12))
    # print(f"reward: {np.sum(reward_data[:, :, episode], axis=0)}")

    # ax.plot(t, np.sum(reward_data[:, :, episode], axis=0))
    # ax.set_title(f"Episode 1")
    # ax.set_xlabel("Rollout")
    # ax.set_ylabel("Reward")
    # ax.grid(True)
    # plt.tight_layout()
    # plt.show()
    # cpg_data.shape[2]
        # plot CPG data
    for rollout in range(num_rollouts):
        total_actions = cpg_data[:, :, rollout, episode]
        t = np.arange(0, rollout_len*dt, dt)
        fig, axs = plt.subplots(nrows=total_actions.shape[0], ncols=1, figsize=(8, 12))
        for j, ax in enumerate(axs):
            ax.plot(t, total_actions[j, :])
            ax.set_title(f"CPG {j + 1}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Radians")
            ax.grid(True)
        plt.tight_layout()
    # plt.show()

    # plot posiiton data
    for rollout in range(num_rollouts):
        total_actions = q_data[:, :, rollout, episode]
        t = np.arange(0, rollout_len*dt, dt)
        fig, axs = plt.subplots(nrows=total_actions.shape[0], ncols=1, figsize=(8, 12))
        for j, ax in enumerate(axs):
            ax.plot(t, total_actions[j, :])
            ax.set_title(f"Motor {j + 1}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Radians")
            ax.grid(True)
        plt.tight_layout()
    plt.show()


    # plot acc data 
    # t = np.arange(0, rollout_len*dt, dt)
    # axos = ["x", "y", "z"]
    # fig, axs = plt.subplots(nrows=acc_data.shape[0], ncols=1, figsize=(8, 12))
    # for j, ax in enumerate(axs):
    #     ax.plot(t, acc_data[j, :, rollout, episode])
    #     ax.set_title(f"Acc Data {axos[j]}")
    #     ax.set_xlabel("Time")
    #     ax.set_ylabel(f"{axos[j]}")
    #     ax.grid(True)
    # plt.tight_layout()
    # plt.show()

    # plot quat data
    # print(f"quat data: {quat_data[:, :, 0, 0]}")
    # t = np.arange(0, rollout_len*dt, dt)
    # axos = ["w", "x", "y", "z"]
    # fig, axs = plt.subplots(nrows=quat_data.shape[0], ncols=1, figsize=(8, 12))
    # for j, ax in enumerate(axs):
    #     ax.plot(t, quat_data[j, :, rollout, episode])
    #     ax.set_title(f"Quat {axos[j]}")
    #     ax.set_xlabel("Time")
    #     ax.set_ylabel(f"{axos[j]}")
    #     ax.grid(True)
    # plt.tight_layout()
    # plt.show()

    # # plot tau data
    # total_torques = tau_data[:, :, rollout, episode]
    # t = np.arange(0, rollout_len*dt, dt)
    # fig, axs = plt.subplots(nrows=total_torques.shape[0], ncols=1, figsize=(8, 12))
    # for j, ax in enumerate(axs):
    #     ax.plot(t, total_torques[j, :])
    #     ax.set_title(f"Motor {j + 1}")
    #     ax.set_xlabel("Time")
    #     ax.set_ylabel("Torque")
    #     ax.grid(True)
    # plt.tight_layout()
    # plt.show()
    
    # plot reward data
    total_rewards = np.sum(reward_data[:, :, episode])
    t = np.arange(0, rollout_len*dt, dt)
    fig, axs = plt.subplots(nrows=total_rewards.shape[0], ncols=1, figsize=(8, 12))
    for j, ax in enumerate(axs):
        ax.plot(t, total_rewards[j, :])
        ax.set_title(f"Motor {j + 1}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Torque")
        ax.grid(True)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()