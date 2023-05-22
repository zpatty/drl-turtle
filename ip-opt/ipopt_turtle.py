import cyipopt
import numpy as np
import scipy.sparse as sps


# System parameters
m = 5
J = None
Jinv = None


total_time = 4.4 # two periods hopefully
dt_dynamics = 0.2
dt_kinematics = 0.1
dt_com_pos = 0.4


# Check that time can be divided by timesteps
assert total_time % dt_dynamics < 0.0000001
assert total_time % dt_kinematics < 0.0000001

# add 1 for endpoint
n_dynamics_timesteps = int(total_time/dt_dynamics)+1
n_kinematics_timesteps = int(total_time/dt_kinematics)+1
n_com_pos_timesteps = int(total_time/dt_com_pos)+1


com_polynomial_variables = 4 # cubic polynomials 
n_r = 3
n_q = 4
state_len = (n_r + n_q) 
n_com_variables = n_com_pos_timesteps*state_len*2 # state and state derivative


n_flippers = 4
n_flipper_p = 3
n_flipper_f = 3

n_flipper_periods = 2
flipper_power_phase_division = 3
flipper_recov_phase_division = flipper_power_phase_division

n_flipper_timesteps = n_flipper_periods*(flipper_power_phase_division+flipper_recov_phase_division) + 1

n_flipper_variables = n_flipper_timesteps*(n_flipper_f+2*n_flipper_p) + (n_flipper_timesteps-1)

n_variables = n_com_variables + n_flippers*n_flipper_variables 


"""
Variables in the form of  
    for each timestep(
        r)
    for each timestep(
        rdot)
    for each timestep(
        q) 
    for each timestep(
        qdot )
    for each flipper ( 
        for each period (
            for each division*2(
                p))
        for each period (
            for each division2*(
                pdot))
        for each period (
            for each division2*(
                f))
        for each period (
            t_power, 
            t_recovery, )
"""

class FlipperVariables:
    def __init__(self, flipper_vars):
        self.p      = flipper_vars[0                               : n_flipper_timesteps*n_flipper_p]
        self.pdot   = flipper_vars[n_flipper_timesteps*n_flipper_p : n_com_pos_timesteps*2*n_flipper_p]
        self.f      = flipper_vars[n_flipper_timesteps*2*n_flipper_p       : n_flipper_timesteps*(2*n_flipper_p+n_flipper_f)]
        self.periods = flipper_vars[-(n_flipper_timesteps-1):]

class StateVariables:
    def __init__(self, x):
        self.r      = x[0                               : n_com_pos_timesteps*n_r]
        self.rdot   = x[n_com_pos_timesteps*n_r         : n_com_pos_timesteps*2*n_r]
        self.q      = x[n_com_pos_timesteps*2*n_r       : n_com_pos_timesteps*(2*n_r+n_q)]
        self.qdot   = x[n_com_pos_timesteps*(2*n_r+n_q) : n_com_pos_timesteps*(2*n_r+2*n_q)]

        self.r = self.r.reshape(n_com_pos_timesteps, n_r)
        self.rdot = self.rdot.reshape(n_com_pos_timesteps, n_r)
        self.q = self.q.reshape(n_com_pos_timesteps, n_q)
        self.qdot = self.qdot.reshape(n_com_pos_timesteps, n_q)

        flipper_vars = x[n_com_variables:]

        self.flipper_1 = FlipperVariables(flipper_vars[0                     : n_flipper_variables])
        self.flipper_2 = FlipperVariables(flipper_vars[n_flipper_variables   : n_flipper_variables*2])
        self.flipper_3 = FlipperVariables(flipper_vars[n_flipper_variables*2 : n_flipper_variables*3])
        self.flipper_4 = FlipperVariables(flipper_vars[n_flipper_variables*3 : n_flipper_variables*4])

    def L(self, q):
        """
        L(q)
        TODO implement
        """
        raise NotImplementedError
    
    def w(self, qdot):
        """
        omega := 2*H^-1*L(q)^-1*qdot
        """
        raise NotImplementedError
    
    def drag(self,t):
        """
        compute the drag at time t
        """
        raise NotImplementedError

    def a2(self,x0,x1,x0dot,x1dot, dt):
        return -(3*(x0-x1)+dt_com_pos*(2*x0dot+x1dot))/dt**2
    
    def a3(self,x0,x1,x0dot,x1dot, dt):
        return (2*(x0-x1)+dt_com_pos*(x0dot+x1dot))/dt**3
        
    def rdotdot(self,t):
        n = t//dt_com_pos
        r0 = self.r[n] 
        r1 = self.r[n+1] 
        r0dot = self.rdot[n] 
        r1dot = self.rdot[n+1] 

        return 2*self.a2(r0,r1,r0dot,r1dot)+6*self.a3(r0,r1,r0dot,r1dot, dt_com_pos)*(t%dt_com_pos)
    
    def flipper_force(self,t,flipper):
        # select flipper
        flipper_vars = self.flipper_1 if flipper == 1 else \
                        (self.flipper_2 if flipper == 2 else \
                        (self.flipper_3 if flipper == 3 else \
                        self.flipper_4))
        
        period_dts = flipper_vars.periods
        period_dt_cumsum = period_dts.cumsum()
        # index into periods
        period_index = np.searchsorted(period_dt_cumsum, t, side="left")
        period_dt = period_dts[period_index]
        
        # TODO double check this line
        phase_division = flipper_power_phase_division if period_index%2==0 else flipper_recov_phase_division
        period_mini_dt = period_dt/phase_division

        

        
    
    def rdotdotRHS(self,t):
        forces = self.drag(t)+self.flipper_force(t,1)+self.flipper_force(t,2)+self.flipper_force(t,3)+self.flipper_force(t,4)
        
        omega = self.w(self.qdot_interp(t))
        velocity = self.v(self.rdot_interp(t))

        return forces/m - np.cross(omega,velocity)


print(n_variables)


# CONSTRAINTS
n_q_start = 4
n_r_start = 3
n_r_end = 3

n_start_end_constraint = n_q_start + n_r_start + n_r_end # TODO maybe add initial conditions for velocity??

n_centroidal_dynamics_constraints = (n_r + n_q)*n_dynamics_timesteps # TODO double check this

n_flipper_kinematic = 1
n_flipper_force = 3 

n_time_constraint = 1

n_flipper_constraints = n_flippers*(n_kinematics_timesteps*n_flipper_kinematic + n_dynamics_timesteps*n_flipper_force + n_time_constraint)

n_constraints = n_flipper_constraints + n_centroidal_dynamics_constraints + n_start_end_constraint

print(n_constraints)

"""
Constraints in the form of  
    for each timestep(
        r)
    for each timestep(
        rdot)
    for each timestep(
        q)
    for each timestep(
        qdot)
    for each flipper ( 
        for each period (
            t_power, 
            for each division(
                p,
                f), 
            t_recovery, 
            for each division(
                p
                ,f)) )
"""



r0 = np.array([0,0,0])
q0 = np.array([0,0,0,0])
r_n = np.array([0,0,0])

class turtle(object):
    def __init__(self):
        """
        No init
        """
        pass

    def objective(self, x):
        """
        The callback for calculating the objective

        No objective function - Return 0
        """
        return 0

    def gradient(self, x):
        """
        The callback for calculating the gradient

        Should be zero as there is no objective
        """
        return np.zeros((len(x),))

    def constraints(self, x):
        """
        The callback for calculating the constraints
        """
        state = StateVariables(x)
        
        start_pos_cons = state.r[0:n_r] - r0
        start_orientation_cons = state.q[0:n_q] - q0
        end_pos_const = state.r[-n_r:] - r_n
        # TODO incomplete

        
        return None

    def jacobian(self, x):
        """
        The callback for calculating the Jacobian

        Derivative of all variables with respect to the constraints
        """
        # TODO incomplete
        return None

    def jacobianstructure(self):
        """
         The structure of the Jacobian
        """
        # TODO incomplete
        return None

    def intermediate(
            self,
            alg_mod,
            iter_count,
            obj_value,
            inf_pr,
            inf_du,
            mu,
            d_norm,
            regularization_size,
            alpha_du,
            alpha_pr,
            ls_trials
            ):

        #
        # Example for the use of the intermediate callback.
        #
        print(f"Objective value at iteration #{iter_count} is - {obj_value}")




nlp = cyipopt.problem(
            n=len(x0),
            m=len(cl),
            problem_obj=hs071(),
            lb=lb,
            ub=ub,
            cl=cl,
            cu=cu
            )

