import numpy as np


def sinusoidal_primitive(t, q, w, A, C, yaw, freq_offset, pitch):
    T = 2 * np.pi / w
    offset = np.pi * (1 - freq_offset/(1 - freq_offset))
    t = np.mod(t, T)
    if t < freq_offset * T:
        q_des1 = C[0] + A[0] * np.cos(w * t / (2 * freq_offset))
        q_des2 = C[1] + A[1] * np.cos(w * t / (2 * freq_offset))
        q_des3 = C[2] + A[2] * np.cos(w * t / (2 * freq_offset))
        qd_des1 = - w / (2 * freq_offset) * A[0] * np.sin(w * t / (2 * freq_offset))
        qd_des2 = - w / (2 * freq_offset) * A[1] * np.sin(w * t / (2 * freq_offset))
        qd_des3 = - w / (2 * freq_offset) * A[2] * np.sin(w * t / (2 * freq_offset))
        qdd_des1 = - (w / (2 * freq_offset))**2 * A[0] * np.cos(w * t / (2 * freq_offset))
        qdd_des2 = - (w / (2 * freq_offset))**2 * A[1] * np.cos(w * t / (2 * freq_offset))
        qdd_des3 = - (w / (2 * freq_offset))**2 * A[2] * np.cos(w * t / (2 * freq_offset))
    else:
        q_des1 = C[0] + A[0] * np.cos(w * t / (2 * (1 - freq_offset)) + offset)
        q_des2 = C[1] + A[1] * np.cos(w * t / (2 * (1 - freq_offset)) + offset)
        q_des3 = C[2] + A[2] * np.cos(w * t / (2 * (1 - freq_offset)) + offset)
        qd_des1 = - w / (2 * (1 - freq_offset)) + A[0] * np.sin(w * t / (2 * (1 - freq_offset)) + offset)
        qd_des2 = - w / (2 * (1 - freq_offset)) + A[1] * np.sin(w * t / (2 * (1 - freq_offset)) + offset)
        qd_des3 = - w / (2 * (1 - freq_offset)) + A[2] * np.sin(w * t / (2 * (1 - freq_offset)) + offset)
        qdd_des1 = - (w / (2 * (1 - freq_offset)))**2 + A[0] * np.cos(w * t / (2 * (1 - freq_offset)) + offset)
        qdd_des2 = - (w / (2 * (1 - freq_offset)))**2 + A[1] * np.cos(w * t / (2 * (1 - freq_offset)) + offset)
        qdd_des3 = - (w / (2 * (1 - freq_offset)))**2 + A[2] * np.cos(w * t / (2 * (1 - freq_offset)) + offset)
    
    qd_rear = (pitch - 0.5) / 0.5 * np.pi / 3
    q_right_des = np.array([q_des1, q_des2, q_des3]).reshape(-1,1)
    qd_right_des = np.array([qd_des1, qd_des2, qd_des3]).reshape(-1,1)
    qdd_right_des = np.array([qdd_des1, qdd_des2, qdd_des3]).reshape(-1,1)
    q_left_des = np.array([q_des1, - q_des2, - q_des3]).reshape(-1,1)
    qd_left_des = np.array([qd_des1, - qd_des2, - qd_des3]).reshape(-1,1)
    qdd_left_des = np.array([qdd_des1, - qdd_des2, - qdd_des3]).reshape(-1,1)
    if yaw > 0.5:
        q_right_des *= (1.0 - yaw) / 0.5
        qd_right_des *= (1.0 - yaw) / 0.5
        qdd_right_des *= (1.0 - yaw) / 0.5
    elif yaw < 0.5:
        q_left_des *= yaw / 0.5
        qd_left_des *= yaw / 0.5
        qdd_left_des *= yaw / 0.5
    q_des = np.vstack((q_right_des, q_left_des, np.array([0.0, qd_rear, 0.0, -qd_rear]).reshape(-1,1)))
    qd_des = np.vstack((qd_right_des, qd_left_des, np.zeros((4,1))))
    qdd_des = np.vstack((qdd_right_des, qdd_left_des, np.zeros((4,1))))
    
    return np.squeeze(q_des), np.squeeze(qd_des), np.squeeze(qdd_des)

def convert_motors_to_q(q, qd):
    q = q.reshape(10,)  - np.pi
    qd = qd.reshape(10,)
    q_new = np.hstack((-q[3:6], [q[0], -q[1], -q[2]], q[6:]))
    qd_new = np.hstack((-qd[3:6], [qd[0], -qd[1], -qd[2]], qd[6:]))
    return q_new, qd_new

def convert_q_to_motors(q, qd, qdd):
    if np.size(q) == 6:
        q = np.hstack((q, np.zeros(4,)))
        qd = np.hstack((qd, np.zeros(4,)))
        qdd = np.hstack((qdd, np.zeros(4,)))
    q_new = np.hstack(([q[3], -q[4], -q[5]], -q[0:3], q[6:])) + np.pi
    qd_new = np.hstack(([qd[3], -qd[4], -qd[5]], -qd[0:3], qd[6:]))
    qdd_new = np.hstack(([qdd[3], -qdd[4], -qdd[5]], -qdd[0:3], qdd[6:]))
    return q_new.reshape(-1,1), qd_new.reshape(-1,1), qdd_new.reshape(-1,1)