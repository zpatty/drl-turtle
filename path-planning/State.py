import numpy as np
from Primitive import Action
from Robot import Robot
from Environment import Environment

class State:
    def __init__(self, r, q, v, w ):
        
        self.r = r.astype(float)
        self.q = q.astype(float)
        self.v = v.astype(float)
        self.w = w.astype(float)

        self.H = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=float) 
        
        self.previous_action = None
        
    
    def copy(self):
        return State(self.r.copy(), self.q.copy(), self.v.copy(), self.w.copy())

    def location(self):
        return self.r
    
    def quaternion(self):
        return self.q
    
    def linear_velocity(self):
        return self.v
    
    def angular_velocity(self):
        return self.w
    
    def skew(self):
        return np.array([[0           , -self.q[3], self.q[2]],
                         [self.q[3], 0            , -self.q[1]],
                        [-self.q[2], self.q[1] , 0]], dtype=float)

    def L(self):
        return np.asarray(np.bmat([[self.q[0:1].reshape(1,1) , -self.q[1:4].reshape(1,3)],
                                   [self.q[1:4].reshape(3,1) , self.q[0]*np.eye(3) + self.skew()]]), dtype=float)


    def R(self):
        return np.asarray(np.bmat([[self.q[0:1].reshape(1,1) , -self.q[1:4].reshape(1,3)],
                                   [self.q[1:4].reshape(3,1) , self.q[0]*np.eye(3) - self.skew()]]), dtype=float)
    
    def A(self):
        result = self.H.T@self.L()@self.R().T@self.H
        return result
    
    def x_dot(self, robot : Robot, F, tau, timestep):
        r_dot =  self.A()@self.v
        q_dot = 0.5*self.L()@self.H@self.w
        v_dot = F/robot.m*timestep + np.cross(self.w, self.v, axisa=0,axisb=0,axisc=0)
        w_dot = robot.J_inv@(tau*timestep - np.cross(self.w,robot.J@self.w, axisa=0,axisb=0,axisc=0))

        return r_dot, q_dot, v_dot, w_dot

    def mini_step(self, F, tau, robot : Robot, timestep):        
        r_dot, q_dot, v_dot, w_dot = self.x_dot(robot, F, tau, timestep)
        self.r += r_dot*timestep
        self.q += q_dot*timestep
        self.v += v_dot*timestep
        self.w += w_dot*timestep
       

    def step(self, action : Action, robot : Robot,  environment : Environment=None, tolerance=0, timestep=0.05, ax=None, color="black"):
        self.previous_action = action
        
        # Power Stroke
        t = 0
        while (t < action.t_power):
            linear_drag = -robot.power_drag_coef*self.v*np.linalg.norm(self.v)
            angular_drag = -robot.angular_drag_coef*self.w

            F = (action.F_finish*t + action.F_catch*(action.t_power-t))/action.t_power
            tau = (action.tau_finish*t + action.tau_catch*(action.t_power-t))/action.t_power

            self.mini_step( F+linear_drag, tau+angular_drag, robot, timestep)
            t+=timestep
            if ax:
                ax.plot(self.r[0], self.r[1], color = color, marker="o", markersize=3)
        
        if environment and not environment.is_location_free(self.r, tolerance):
            return False


        # Recovery Stroke
        t=0
        while (t < action.t_recovery):
            linear_drag = -robot.recovery_drag_coef*self.v*np.linalg.norm(self.v)
            angular_drag = -robot.angular_drag_coef*self.w
            
            self.mini_step(linear_drag, angular_drag, robot, timestep)
            t+=timestep
            if ax:
                ax.plot(self.r[0], self.r[1], color = color, marker="o", markersize=3)
        
        if environment and not environment.is_location_free(self.r, tolerance):
            return False
        
        if ax:
            ax.plot(self.r[0], self.r[1], marker="o", markersize=7, markeredgecolor=color, markerfacecolor="blue")
        return True

        

    def step_new(self, action : Action, robot : Robot, environment : Environment = None, tolerance=0,  ax=None, color="black"):
        next_state = self.copy()
        successful_step = next_state.step(action, robot, environment, tolerance, ax=ax, color=color)
        return next_state if successful_step else None


if __name__=="__main__":

    robot = Robot()
    
    initial_r = np.array([0,5,0])
    initial_q = np.array([1,0,0,0])
    initial_v = np.array([1,0,0]) # 1m/s is actually a good estimate for sea turtle speeds
    initial_w = np.array([0,0,0])

    initial_state = State(initial_r, initial_q, initial_v, initial_w)

    torque = 1.5
    t_power = 0.75
    t_recovery = 1.5

    p_forward  = Action(t_power, t_recovery, np.array([75,0,0]), np.array([50,0,0]), np.array([0,0,0]), np.array([0,0,0]) )
    
    p_left     = Action(t_power, t_recovery, np.array([60,0,0]), np.array([50,0,0]), np.array([0,0, torque]), np.array([0,0,0]) )
    p_right    = Action(t_power, t_recovery, np.array([60,0,0]), np.array([50,0,0]), np.array([0,0,-torque]), np.array([0,0,0]) )

    action_list = [p_left]

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.set_xlim((-4,7))
    ax.set_ylim((3,14))


    next_state = initial_state.step_new(p_left, robot, ax=ax)
    for _ in range(7):
        next_state.step(p_left, robot, ax=ax)

    plt.show()
