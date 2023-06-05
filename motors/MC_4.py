########################################################
# This function calculates the Mass and Coriolis + Gravity Matrices
# given the current state and geometric parameters of the pushpuppet robot
########################################################

def MC(q,qd,m,mm,hm,rm,r,L0,d,N):
    # q = state
    # qd = velocity
    # qdd = acceleration
    # m = mass of a module;
    # mm = mass of a motor;
    # hm = height of a motor;
    # rm = radius a motor;
    # r = radius of the hex plate;
    # L0 = Length of a module;
    # d = distance to cable;
    # N = number of links (number of modules plus number of motors)
    
    a_grav = np.array([0, 0, -9.81, 0, 0, 0]).reshape(-1, 1)
    
    qi = {0: [0],
          1: [1,4],
          2: [4],
          3: [5,8],
          4: [8],
          5: [9,12]
          6: [12], 
          7: [13,16]}
    
    
    inertia = [1/4*m*r**2,  1/4*m*r**2,  1/2*m*r**2]
    I = dict()
    I[0] = np.diag([m, m, m, 1/4*m*r**2,  1/4*m*r**2,  1/2*m*r**2])
    I[1] = I[0]
    I[3] = I[0]
    I[5] = I[0]
    I[7] = I[0]
    
    inertia_motor = np.array([1/4*mm*rm**2, 1/4*mm*rm**2, 1/2*mm*rm**2]).reshape(-1,1)
    c = skew(np.array([0, 0, hm/2]).reshape(-1,1))
    I[2] = np.zeros((6,6))
    I[2][:3, :3] = mm*np.identity(3)
    I[2][:3, 3:] = mm*c.T
    I[2][3:, :3] = mm*c
    I[2][3:, 3:] = inertia_motor + mm*(c@c.T)
    I[4] = I[2]
    I[6] = I[2]
    
    S = dict()
    Xup = dict()
    v = dict()
    a = dict()
    f = dict()

    Xtree = 1     
    i = 0
    g = np.array([[cos(q[qi[i]][0]), -sin(q[qi[i]][0]), 0, 0], [sin(q[qi[i]][0]), cos(q[qi[i]][0]), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    XJ = adj_calc(g,1)
    S[i] = np.array([0, 0, 0, 0, 0, 1]).reshape(-1,1)
    Xup[i] = XJ*Xtree
    vJ = S[i]*qd[qi[i]]
    v[i] = vJ
    a[i] = Xup[i].dot(-a_grav) + spatial_cross(v[i]).dot(vJ)
    
    f[i] = I[i].dot(a[i]) + -np.matmul(spatial_cross(v[i]).T, I[i]).dot(v[i])

    # Recursive Newton Euler to Calculate C+G
    for i in range(1,N):

      if (i%2 != 0):
          qa,qb = qi[i]
          [XJ, S[i],_,dJ] = PCC_jacobian(q[qa:qb],d,L0,qd[qa:qb])
          if type(Xtree) != int:
            Xup[i] = XJ@Xtree
          else:
            Xup[i] = XJ*Xtree
          Xtree = np.identity(6)
      else:
          g = np.array([[cos(q[qi[i]][0]),  0,  sin(q[qi[i]][0]),  0],
                        [0, 1, 0, 0],
                        [-sin(q[qi[i]][0]), 0, cos(q[qi[i]][0]), 0],
                        [0, 0, 0, 1]])
          XJ = adj_calc(g,1)
          S[i] = np.array([0,0,0,0,1,0]).reshape(-1,1)
          dJ = np.zeros((1,1))
          Xup[i] = np.matmul(XJ,Xtree)
          Xtree = adj_calc(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, hm], [0, 0, 0, 1]]),1)
      
      if len(qi[i]) > 1:
        qa,qb = qi[i]
        vJ = S[i].dot(qd[qa:qb])
        c1 = Xup[i].dot(a[i-1])
        c2 = dJ.dot(qd[qa:qb])

        v[i] = Xup[i].dot(v[i-1]) + vJ
        a[i] = Xup[i].dot(a[i-1]) + dJ.dot(qd[qa:qb]) + spatial_cross(v[i]).dot(vJ)
      else:
        vJ = S[i].dot(qd[qi[i]])
        v[i] = Xup[i].dot(v[i-1]) + vJ
        a[i] = Xup[i].dot(a[i-1]) + dJ.dot(qd[qi[i]]) + spatial_cross(v[i]).dot(vJ)
      f[i] = I[i].dot(a[i]) + -np.matmul(spatial_cross(v[i]).T, I[i]).dot(v[i])

    C = np.zeros((16,1))  
  
    for i in range(N-1, -1, -1):
      if len(qi[i]) > 1:
        qa, qb = qi[i]
        C[qa:qb] = 2 * S[i].T.dot(f[i])
      else:
        C[qi[i]] = 2 * S[i].T.dot(f[i])
      if i != 0:
      
        f[i-1] = f[i-1] + Xup[i].T.dot(f[i])
    
    # % Composite Rigid Body Algorithm to calculate M
    IC = I				# composite inertia calculation
    for i in range(N-1, -1, -1):
      if i != 0:
        IC[i-1] = IC[i-1] + Xup[i].T@IC[i]@Xup[i]
    H = np.zeros((16, 16))
    for i in range(0,N):
      #LINES 136-END of Matlab code
      if (S[i].shape[1]) > 1:
        fh = IC[i]@S[i]
      else:
        fh = IC[i].dot(S[i])
      if len(qi[i]) > 1:
        qa, qb = qi[i]
        H[qa:qb,qa:qb] = S[i].T @ fh
      else:
         H[qi[i],qi[i]] = S[i].T.dot(fh)   
       
      j = i
      while (j < N and j > 0):
        if (fh.shape[1]) > 1:
          fh = Xup[j].T @ fh
        else:
          fh = Xup[j].T.dot(fh)
        
        j = j - 1
        if len(qi[i]) > 1:
          qa, qb = qi[i]
          if len(qi[j]) > 1:
            qaj, qbj = qi[j]
            H[qa:qb,qaj:qbj] = fh.T @ S[j] 
            H[qaj:qbj,qa:qb] = S[j].T @ fh

          else:
            qj = qi[j]
            H[qa:qb,qj] = fh.T @ S[j] 
            H[qj,qa:qb] = S[j].T @ fh

        else:
          
          if len(qi[j]) > 1:
            qa, qb = qi[j]
            H[qi[i],qa:qb] = np.matmul(fh.T,S[j]) 
            H[qa:qb,qi[i]] = S[j].T @ fh

          else:
            H[qi[i],qi[j]] = np.matmul(fh.T,S[j]) 
            H[qi[j],qi[i]] = S[j].T.dot(fh)
    
    M = H
  return M, C