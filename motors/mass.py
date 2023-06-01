def mass(q, dq):
    q2 = q[2]
    q3 = q[3]

    dq1 = q[1]
    dq2 = q[2]
    dq3 = q[3]

    I11 = I[1]
    I12 = I[2]
    I13 = I[3]
    I21 = I[4]
    I22 = I[5]
    I23 = I[6]
    I31 = I[7]
    I32 = I[8]
    I33 = I[9]
    I41 = I[10]
    I42 = I[11]
    I43 = I[12]
    l1 = lengths[1]
    l2 = lengths[2]
    l3 = lengths[3]
    lo1 = lengths[4]
    lo2 = lengths[5]
    lo3 = lengths[6]
    lo4 = lengths[7]
    lf = lengths[8]
    wf = lengths[9]
    ls = lengths[10]
    lm = lengths[11]
    m1 = m[1]
    m2 = m[2]
    m3 = m[3]
    m4 = m[4]
    
    M = np.array([[I21 + I32 + I42 + l2**2*m2 + l2**2*m3 + l2**2*m4 + (lf**2*m4)/4 + lo2**2*m3 + lo2**2*m4 + lo3**2*m3 + lo3**2*m4 + lo4**2*m4 + I31*cos(q2)**2 - I32*cos(q2)**2 + (I41*cos(q2)**2)/2 - I42*cos(q2)**2 + (I43*cos(q2)**2)/2 + (I41*cos(2*q3)*cos(q2)**2)/2 - (I43*cos(2*q3)*cos(q2)**2)/2 + lf*lo2*m4 + lf*lo4*m4 + 2*lo2*lo4*m4 - (lf**2*m4*cos(q2)**2)/4 - lo2**2*m3*cos(q2)**2 - lo2**2*m4*cos(q2)**2 - lo4**2*m4*cos(q2)**2 + (m4*wf**2*cos(q2)**2)/4 + l2*m4*wf*cos(q2) - l2*lf*m4*sin(q2) - 2*l2*lo2*m3*sin(q2) - 2*l2*lo2*m4*sin(q2) - 2*l2*lo4*m4*sin(q2) - lf*lo2*m4*cos(q2)**2 - lf*lo4*m4*cos(q2)**2 - 2*lo2*lo4*m4*cos(q2)**2 - (lf*m4*wf*sin(2*q2))/4 - (lo2*m4*wf*sin(2*q2))/2 - (lo4*m4*wf*sin(2*q2))/2, (I41*sin(2*q3)*cos(q2))/2 - (I43*sin(2*q3)*cos(q2))/2 + (lf*lo3*m4*cos(q2))/2 + lo2*lo3*m3*cos(q2) + lo2*lo3*m4*cos(q2) + lo3*lo4*m4*cos(q2) + (lo3*m4*wf*sin(q2))/2, I42*sin(q2)]
                [                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  (I41*sin(2*q3)*cos(q2))/2 - (I43*sin(2*q3)*cos(q2))/2 + (lf*lo3*m4*cos(q2))/2 + lo2*lo3*m3*cos(q2) + lo2*lo3*m4*cos(q2) + lo3*lo4*m4*cos(q2) + (lo3*m4*wf*sin(q2))/2,      I33 + I41/2 + I43/2 + (lf**2*m4)/4 + lo2**2*m3 + lo2**2*m4 + lo4**2*m4 + (m4*wf**2)/4 - (I41*cos(2*q3))/2 + (I43*cos(2*q3))/2 + lf*lo2*m4 + lf*lo4*m4 + 2*lo2*lo4*m4,           0]
                [                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           I42*sin(q2),                                                                                                                                                                    0,         I42]])

    C = np.array([[        - dq2*(2*I31*cos(q2)*sin(q2) - 2*I32*cos(q2)*sin(q2) + I41*cos(q2)*sin(q2) - 2*I42*cos(q2)*sin(q2) + I43*cos(q2)*sin(q2) - (lf**2*m4*cos(q2)*sin(q2))/2 - 2*lo2**2*m3*cos(q2)*sin(q2) - 2*lo2**2*m4*cos(q2)*sin(q2) - 2*lo4**2*m4*cos(q2)*sin(q2) + (m4*wf**2*cos(q2)*sin(q2))/2 + l2*lf*m4*cos(q2) + 2*l2*lo2*m3*cos(q2) + 2*l2*lo2*m4*cos(q2) + 2*l2*lo4*m4*cos(q2) + l2*m4*wf*sin(q2) + I41*cos(2*q3)*cos(q2)*sin(q2) - I43*cos(2*q3)*cos(q2)*sin(q2) + (lf*m4*wf*cos(2*q2))/2 + lo2*m4*wf*cos(2*q2) + lo4*m4*wf*cos(2*q2) - 2*lf*lo2*m4*cos(q2)*sin(q2) - 2*lf*lo4*m4*cos(q2)*sin(q2) - 4*lo2*lo4*m4*cos(q2)*sin(q2)) - dq3*(I41*sin(2*q3)*cos(q2)**2 - I43*sin(2*q3)*cos(q2)**2), dq3*(I42*cos(q2) + I41*cos(2*q3)*cos(q2) - I43*cos(2*q3)*cos(q2)) - dq1*(2*I31*cos(q2)*sin(q2) - 2*I32*cos(q2)*sin(q2) + I41*cos(q2)*sin(q2) - 2*I42*cos(q2)*sin(q2) + I43*cos(q2)*sin(q2) - (lf**2*m4*cos(q2)*sin(q2))/2 - 2*lo2**2*m3*cos(q2)*sin(q2) - 2*lo2**2*m4*cos(q2)*sin(q2) - 2*lo4**2*m4*cos(q2)*sin(q2) + (m4*wf**2*cos(q2)*sin(q2))/2 + l2*lf*m4*cos(q2) + 2*l2*lo2*m3*cos(q2) + 2*l2*lo2*m4*cos(q2) + 2*l2*lo4*m4*cos(q2) + l2*m4*wf*sin(q2) + I41*cos(2*q3)*cos(q2)*sin(q2) - I43*cos(2*q3)*cos(q2)*sin(q2) + (lf*m4*wf*cos(2*q2))/2 + lo2*m4*wf*cos(2*q2) + lo4*m4*wf*cos(2*q2) - 2*lf*lo2*m4*cos(q2)*sin(q2) - 2*lf*lo4*m4*cos(q2)*sin(q2) - 4*lo2*lo4*m4*cos(q2)*sin(q2)) - dq2*(I41*sin(2*q3)*sin(q2) - I43*sin(2*q3)*sin(q2) - lo3*m4*wf*cos(q2) + lf*lo3*m4*sin(q2) + 2*lo2*lo3*m3*sin(q2) + 2*lo2*lo3*m4*sin(q2) + 2*lo3*lo4*m4*sin(q2)), dq2*(I42*cos(q2) + I41*cos(2*q3)*cos(q2) - I43*cos(2*q3)*cos(q2)) - dq1*(I41*sin(2*q3)*cos(q2)**2 - I43*sin(2*q3)*cos(q2)**2)]
    [dq1*(2*I31*cos(q2)*sin(q2) - 2*I32*cos(q2)*sin(q2) + I41*cos(q2)*sin(q2) - 2*I42*cos(q2)*sin(q2) + I43*cos(q2)*sin(q2) - (lf**2*m4*cos(q2)*sin(q2))/2 - 2*lo2**2*m3*cos(q2)*sin(q2) - 2*lo2**2*m4*cos(q2)*sin(q2) - 2*lo4**2*m4*cos(q2)*sin(q2) + (m4*wf**2*cos(q2)*sin(q2))/2 + l2*lf*m4*cos(q2) + 2*l2*lo2*m3*cos(q2) + 2*l2*lo2*m4*cos(q2) + 2*l2*lo4*m4*cos(q2) + l2*m4*wf*sin(q2) + I41*cos(2*q3)*cos(q2)*sin(q2) - I43*cos(2*q3)*cos(q2)*sin(q2) + (lf*m4*wf*cos(2*q2))/2 + lo2*m4*wf*cos(2*q2) + lo4*m4*wf*cos(2*q2) - 2*lf*lo2*m4*cos(q2)*sin(q2) - 2*lf*lo4*m4*cos(q2)*sin(q2) - 4*lo2*lo4*m4*cos(q2)*sin(q2)) - dq3*(I42*cos(q2) - I41*cos(2*q3)*cos(q2) + I43*cos(2*q3)*cos(q2)),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       dq3*(I41*sin(2*q3) - I43*sin(2*q3)),                     dq2*(I41*sin(2*q3) - I43*sin(2*q3)) - dq1*(I42*cos(q2) - I41*cos(2*q3)*cos(q2) + I43*cos(2*q3)*cos(q2))]
    [                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           dq2*(I42*cos(q2) - I41*cos(2*q3)*cos(q2) + I43*cos(2*q3)*cos(q2)) + dq1*(I41*sin(2*q3)*cos(q2)**2 - I43*sin(2*q3)*cos(q2)**2),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   dq1*(I42*cos(q2) - I41*cos(2*q3)*cos(q2) + I43*cos(2*q3)*cos(q2)) - dq2*(I41*sin(2*q3) - I43*sin(2*q3)),                                                                                                                           0]])
    
    return M, C