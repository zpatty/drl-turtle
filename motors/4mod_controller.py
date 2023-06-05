import numpy as np
from numpy import log, exp
from math import sqrt, sin, cos
from MC_4 import MC
from controller_functions import repmat
from scipy.linalg import block_diag
def puppet_controller(q,dq,qd,dqd,ddqd,d,hp,m,mm,hm,rm,r,N,kb,ks,bb,bs,L0,e,Kp,KD,kc,Ka,c_offset,Lm=0.013,contact=1):
    
    qp = dict()
    qp[0] = q[1:4]
    qp[1] = q[5:8]
    qp[2] = q[9:12]
    qp[3] = q[13:16]
    qdp = dict()
    qdp[0] = qd[1:4]
    qdp[1] = qd[5:8]
    qdp[2] = qd[9:12]
    qdp[3] = qd[13:16]
    dqp = dict()
    dqp[0] = dq[1:4]
    dqp[1] = dq[5:8]
    dqp[2] = dq[9:12]
    dqp[3] = dq[13:16]
    

    [M, C] = MC(q,dq,m,mm,hm,rm,r,L0,d,N)
    kbrep = repmat(np.array([kb,kb,ks,0]).reshape(-1,1,),2,1)
    bbrep = repmat(np.array([bb,bb,bs,2]).reshape(-1,1,),2,1)
    kbrep = np.ndarray.tolist(kbrep.T[0])
    bbrep = np.ndarray.tolist(bbrep.T[0])
    karr = [0] + kbrep + [kb,kb,ks]
    darr = [0] + bbrep + [bb,bb,bs]
    K = np.diag(karr)
    D = np.diag(darr)
    
    Fci = dict()
    Ai = dict()
    c_arr = np.zeros((len(qp), 1))
    Kp_ind = dict()
    Kp_ind[0] = Kp[1:4,1:4]
    Kp_ind[1] = Kp[5:8,5:8]
    Kp_ind[2] = Kp[9:12,9:12]
    K_ind = dict()
    K_ind[0] = K[1:4,1:4]
    K_ind[1] = K[5:8,5:8]
    K_ind[2] = K[9:12,9:12]
    for i in range(0,len(qp)):
        dx = qp[i][0][0]
        dy = qp[i][1][0]
        dL = qp[i][2][0]
        delta = sqrt(dx**2+dy**2)
        Dq = delta - sin(delta)
        L = dL + L0 - Lm
        theta = delta/d
        if theta < 1e-6:
            c = (L)/3
            dtheta_dx = 0
            dtheta_dy = 0
            dc_dx = dtheta_dx
            dc_dy = dtheta_dy
            dc_dL = 1/3
        else:
            c = 2*((L0 + dL)/(theta) - d)*sin(theta/6)
            dc_dtheta = - (cos(theta/6)*(2*d - (2*(L))/theta))/6 - (2*sin(theta/6)*(L))/theta**2
            dtheta_dx = dx/delta/d
            dtheta_dy = dy/delta/d
            dc_dx = dc_dtheta*dtheta_dx
            dc_dy = dc_dtheta*dtheta_dy
            dc_dL = 2/theta*sin(theta/6)
        c_arr[i][0] = c
        gain = K_ind[i].dot(abs(qdp[i])) * Ka
        sig = exp(-kc*(c+c_offset))/(exp(-kc*(c+c_offset)) + 1)

        if dx < 10e-6 and dy < 10e-6:
            Fcu = 0
            Fcv = 0    
            Aq = np.array([[0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]])
        else:
            Fcu = exp(-kc*c)/(exp(-kc*c) + 1) * (gain[0][0])*2*dc_dx*3
            Fcv = exp(-kc*c)/(exp(-kc*c) + 1) * (gain[1][0])*2*dc_dy*3
            Aq = np.array([[dx * dy * Dq / delta**3, (- dx**2 * delta - dy**2 * sin(delta)) / delta**3, dx * Dq * L / delta**3],
            [(dy**2 * delta + dx**2 * sin(delta)) / delta**3, - dx * dy * Dq / delta**3, dy * Dq * L / delta**3],
            [0, 0, sin(delta) / delta]])
        Fch = exp(-kc*c)/(exp(-kc*c) + 1) * (gain[2][0])*dc_dL*3
        Fci[i] = [Fcu, Fcv, Fch]
        term = (-sig * Ka * (K_ind[i].dot(qdp[i]) + Kp_ind[i].dot((qdp[i]-qp[i])))).tolist()
        # print(f"term: {term[0]}\n")
        Fci[i] = [term[0][0], term[1][0], term[2][0]]
        # print(f"term: {Fci[i]}")
        Al = np.array([[d*cos(30 * np.pi/180),d*cos(30 * np.pi/180),-d],
        [-d*cos(60*np.pi/180),d*cos(60*np.pi/180),0],
        [1, 1, 1]])

        At = np.array([[-1,0,0], [0,-1,0],[0,0,-1]])
        # Ai[i] = At @ Aq @ Al
        Ai[i] = np.matmul(Aq, np.matmul(At, Al))
    fcarr = [0] +  Fci[0] + [0] + Fci[1] +  [0] +  Fci[2]
    Fc = np.array(fcarr).reshape(-1,1)
    ones = np.ones((1,1))
    A = block_diag(ones, Ai[0], ones, Ai[1], ones, Ai[2])
    # print(f"[CNTRLLR] C: {C}\n")
    # print(f"[CNTRLLR] M: {M.dot(ddqd)}\n")
    # print(f"[CNTRLLR] K: {K.dot(qd)}\n")
    # print(f"[CNTRLLR] D: {D.dot(dqd)}\n")
    # print(f"[CNTRLLR] Fc: {Fc}\n")
    # print(f"[CNTRLLR] Kp: {Kp.dot((qd-q))}\n")
    # print(f"[CNTRLLR] D: {KD*(dqd - dq)}\n")
    GG = (C + M.dot(ddqd) + K.dot(qd) + D.dot(dqd) + (Fc*contact) + Kp.dot((qd-q)) + KD*(dqd - dq))

    GG = np.float64(GG)
    # print(f"GG: {GG}\n")
    tau = np.linalg.solve(A,GG)
    return tau, c_arr
