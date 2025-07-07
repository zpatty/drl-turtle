import numpy as np

def crawl(t):
    s = t % 4.3 / 4.3
    q = np.zeros(10,)
    if s < 0.25:
        q_roll = 30
        q_yaw = 45
    elif s >= 0.25 and s < 0.5:
        q_roll = -30
        q_yaw = 45
    elif s >= 0.5:
        q_roll = -30
        q_yaw = -45
    # print(s)
    
    q[0], q[3] = q_roll, q_roll
    q[1], q[4] = q_yaw, q_yaw
    
    return q * np.pi / 180
