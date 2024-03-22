import os
import sys
import scipy
import numpy as np
# from EPHE import EPHE
# from pgpelib import PGPE
# from pgpelib.policies import LinearPolicy, MLPPolicy
# from pgpelib.restore import to_torch_module
# import matplotlib.pyplot as plt

submodule = os.path.expanduser("~") + "/drl-turtle/ros2_ws/src/turtle_rl/turtle_rl"
sys.path.append(submodule)

class AukeCPG:
    """
    Auke Ijspeert implementation (the coupled CPG model)
    paper references:
        : https://link.springer.com/article/10.1007/s10514-007-9071-6

    This implementation was adapted for the sea turtle structure, 
    where number of coupled oscillators depended on number of motors on fins:
        : front fins - 3 coupled oscillators per fin
        : back fins - 2 coupled oscillators per fin

    Auke's CPG model technically outputs position in radians, so this most likely will be 
    used in tandem with the PD method of turtle CPG learning. 
    """
    def __init__(self, 
                 num_params=21,
                 num_mods=10,
                 phi=0.0,
                 w=np.random.rand() * np.pi * 2,
                 a_r=20,
                 a_x=20,
                 dt=0.05):

        self.w = w                          # coupled weight bias
        self.phi = phi                      # coupled phase bias- this seems to always be 0 according to auke
        self.num_mods = num_mods            # number of CPG modules
        self.theta = np.zeros((num_mods))   # oscillating setpoint of oscillator i (in radians)

        self.params = np.random.rand((num_params)) * 0.1        # holds the params of the CPG [omega, R, X] = [freq, amplitude, offset]
        self.PHI = np.zeros((num_mods))                         # phase state variables (radians)
        self.r = np.zeros((num_mods))                           # amplitude state variables (radians)
        self.x = np.zeros((num_mods))                           # offset state variables (radians)                           
        self.v = np.zeros((num_mods))                           # to handle second order eq of amplitude
        self.m = np.zeros((num_mods))                           # to handle second order eq of offset
        self.dt = dt
        self.a_r = a_r
        self.a_x = a_x

        print(f"starting phi: {self.phi}\n")
        print(f"starting w: {self.w}\n")
        print(f"number of CPG modules: {self.num_mods}\n")
    def set_parameters(self, params):
        """
        Updates parameters of the CPG oscillators.
        We currently have the structure to be a 21x1 vector like so:
        = omega: frequency for all oscillators
        = R1 : amplitude for CPG mod 2
        = ...       
        = Rn: amplitude for CPG mod n
        = X1 : offset for CPG mod 1
        = ...
        = Xn: offset for CPG mod n
        """
        self.params = params
        # print(f"current params: {self.params}")
    
    def get_params(self):
        return self.params
    
    def reset(self):
        """
        Reset your CPGs?
        """
        self.PHI = np.random.uniform(low=0.01, high=3, size=self.num_mods)
        self.r = np.random.uniform(low=0.01, high=3, size=self.num_mods)
        self.x = np.random.uniform(low=0.01, high=3, size=self.num_mods)
        self.m = np.random.uniform(low=0.01, high=3, size=self.num_mods)
        self.v = np.random.uniform(low=0.01, high=3, size=self.num_mods)

        
    def get_coupled_action(self, dt):
        """
        Return action based off of observation and current CPG params
        
        """
        def ode_fin(state, omega, R, X, rs, phis):
            PHI, r, x, v, m= state
            dPhidt = omega
            for i in range(len(len(rs))):
                r_other = rs[i]
                phi_other = phis[i]
                dPhidt += self.w*r_other*np.sin(phi_other - PHI - self.phi)
            dRdt = v
            dVdt = self.a_r * ((self.a_r/4) * (R-r) - dRdt)
            dXdt = m
            dMdt = self.a_x * ((self.a_x/4) * (X-x) - dXdt)
            return [dPhidt, dRdt, dXdt, dVdt, dMdt]
        # calculate y_out, which in this case we are using as the tau we pass into the turtle
        action = np.zeros((self.num_mods))
        omega = self.params[0]
        Rs = self.params[1:self.num_mods + 1]
        Xs = self.params[self.num_mods + 1:]
        front_fins = [[0, 1, 2], [3, 4, 5]]
        back_fins = [[6, 7], [8, 9]]
        # for every front fin
        for fin in front_fins:
            num_coupled = 3
            # for each oscillator in front fin
            for f in range(len(fin)):
                # grab the index of the oscillator
                idx = fin[f]
                R = Rs[idx]
                X = Xs[idx]
                # find indices of the other two oscillators coupled to current oscillator
                ind1 = fin[(f + 1)%num_coupled]
                ind2 = fin[(f + 2)%num_coupled]
                rs = [self.r[ind1] , self.r[ind2]]
                phis = [self.PHI[ind1] , self.PHI[ind2]]
                # state of current oscillator
                state = [self.PHI[idx], self.r[idx], self.x[idx], self.v[idx], self.m[idx]]
                t_points = [0,self.dt]
                solution = scipy.integrate.solve_ivp(
                    fun = lambda t, y: ode_fin(state, omega, R, X, rs, phis),
                    t_span=t_points, 
                    y0=state,
                    method='RK45',
                    t_eval = t_points
                )
                try:
                    self.PHI[idx] = solution.y[0, 1]
                    self.r[idx] = solution.y[1, 1]
                    self.x[idx] = solution.y[2, 1]
                    self.v[idx] = solution.y[3, 1]
                    self.m[idx] = solution.y[4, 1]
                except:
                    print("failed to solve ode with the following: \n")
                    print(f"state= {state}\n")
                    print(f"dt= {dt}\n")
                    pass
                # grab output of oscillator i
                self.theta[idx] = self.x[idx] + self.r[idx] * np.cos(self.PHI[idx])
                action[idx] = self.theta[idx]

        # for every back fin
        for fin in back_fins:
            num_coupled = 2
            for f in range(len(fin)):
                idx = fin[f]                            # grab index of current oscillator
                ind = fin[(f + 1) % num_coupled]        # grab index of other oscillator
                rs = [self.r[ind]]
                phis = [self.PHI[ind]]
                # state of current oscillator
                state = [self.PHI[idx], self.r[idx], self.x[idx], self.v[idx], self.m[idx]]
                t_points = [0,self.dt]
                solution = scipy.integrate.solve_ivp(
                    fun = lambda t, y: ode_fin(state, omega, R, X, rs, phis),
                    t_span=t_points, 
                    y0=state,
                    method='RK45',
                    t_eval = t_points
                )
                try:
                    self.PHI[idx] = solution.y[0, 1]
                    self.r[idx] = solution.y[1, 1]
                    self.x[idx] = solution.y[2, 1]
                    self.v[idx] = solution.y[3, 1]
                    self.m[idx] = solution.y[4, 1]
                except:
                    print("failed to solve ode with the following: \n")
                    print(f"state= {state}\n")
                    print(f"dt= {dt}\n")
                    pass
                # grab output of oscillator i
                self.theta[idx] = self.x[idx] + self.r[idx] * np.cos(self.PHI[idx])
                action[idx] = self.theta[idx]
        return action

    def get_action(self):
        """
        Get single time step action from CPGs
        """
        omega = self.params[0]
        Rs = self.params[1:self.num_mods + 1]
        Xs = self.params[self.num_mods + 1:]

        action = np.zeros((self.num_mods))
        # for every front fin
        for m in range(self.num_mods):
            # print(f"mod {m + 1}")
            # for every CPG module calculate output for each oscillator
            R = Rs[m]
            X = Xs[m]
            # find indices of the other two oscillators coupled to current oscillator state of current oscillator
            state = [self.PHI[m], self.r[m], self.x[m], self.v[m], self.m[m]]
            t_points = [0, self.dt]
            solution = scipy.integrate.solve_ivp(
                fun = lambda t, y: self.ode_fin(state, omega, R, X),
                t_span=t_points, 
                y0=state,
                method='RK45',
                t_eval = t_points
            )
            try:
                self.PHI[m] = solution.y[0, 1]
                self.r[m] = solution.y[1, 1]
                self.x[m] = solution.y[2, 1]
                self.v[m] = solution.y[3, 1]
                self.m[m] = solution.y[4, 1]
            except:
                print("failed to solve ode with the following: \n")
                print(f"state= {state}\n")
                print(f"dt= {self.dt}\n")
                pass
            self.theta[m] = self.x[m] + self.r[m] * np.cos(self.PHI[m])
            # grab output of oscillator i
            if m in [3, 4, 5, 8, 9]:
                self.theta[m] = -1 * self.theta[m]
            self.theta[m] += np.pi
            action[m] = self.theta[m]  
        return action          
    def ode_fin(self, state, omega, R, X):
        PHI, r, x, v, m= state
        dPhidt = omega
        dRdt = v
        dVdt = self.a_r * ((self.a_r/4) * (R-r) - dRdt)
        dXdt = m
        dMdt = self.a_x * ((self.a_x/4) * (X-x) - dXdt)
        return [dPhidt, dRdt, dXdt, dVdt, dMdt]

    def get_rollout(self, episode_length=60):
        """
        Calculate the rollout for an uncoupled oscillator framework
        """
        print("GETTING ROLLOUT")
        # calculate y_out, which in this case we are using as the tau we pass into the turtle
        total_actions = np.zeros((self.num_mods, episode_length))
        omega = self.params[0]
        Rs = self.params[1:self.num_mods + 1]
        Xs = self.params[self.num_mods + 1:]
        for i in range(episode_length):
            print(f"time step: {i}")
            action = np.zeros((self.num_mods))
            # for every front fin
            for m in range(self.num_mods):
                # print(f"mod {m + 1}")
                # for every CPG module calculate output for each oscillator
                R = Rs[m]
                X = Xs[m]
                # find indices of the other two oscillators coupled to current oscillator state of current oscillator
                state = [self.PHI[m], self.r[m], self.x[m], self.v[m], self.m[m]]
                t_points = [0, self.dt]
                solution = scipy.integrate.solve_ivp(
                    fun = lambda t, y: self.ode_fin(state, omega, R, X),
                    t_span=t_points, 
                    y0=state,
                    method='RK45',
                    t_eval = t_points
                )
                try:
                    self.PHI[m] = solution.y[0, 1]
                    self.r[m] = solution.y[1, 1]
                    self.x[m] = solution.y[2, 1]
                    self.v[m] = solution.y[3, 1]
                    self.m[m] = solution.y[4, 1]
                except:
                    print("failed to solve ode with the following: \n")
                    print(f"state= {state}\n")
                    print(f"dt= {self.dt}\n")
                    pass
                self.theta[m] = self.x[m] + self.r[m] * np.cos(self.PHI[m])
                # grab output of oscillator i
                if m in [3, 4, 5, 8, 9]:
                    self.theta[m] = -1 * self.theta[m]
                self.theta[m] += np.pi/2
                action[m] = self.theta[m]            
            # record action for time step into total_actions struct
            total_actions[:, i] = action

        return total_actions

    def get_coupled_rollout(self, episode_length=60):
        """
        Calculate the entire rollout
        """
        def ode_fin(state, omega, R, X, rs, phis):
            PHI, r, x, v, m= state
            dPhidt = omega
            for i in range(len(rs)):
                r_other = rs[i]
                phi_other = phis[i]
                dPhidt += self.w*r_other*np.sin(phi_other - PHI - self.phi)
            dRdt = v
            dVdt = self.a_r * ((self.a_r/4) * (R-r) - dRdt)
            dXdt = m
            dMdt = self.a_x * ((self.a_x/4) * (X-x) - dXdt)
            return [dPhidt, dRdt, dXdt, dVdt, dMdt]
        # calculate y_out, which in this case we are using as the tau we pass into the turtle
        total_actions = np.zeros((self.num_mods, episode_length))
        omega = self.params[0]
        Rs = self.params[1:self.num_mods + 1]
        Xs = self.params[self.num_mods + 1:]
        front_fins = [[0, 1, 2], [3, 4, 5]]
        back_fins = [[6, 7], [8, 9]]
        for i in range(episode_length):
            action = np.zeros((self.num_mods))
            # for every front fin
            for fin in front_fins:
                num_coupled = 3
                # for each oscillator in front fin
                for f in range(len(fin)):
                    # grab the index of the oscillator
                    idx = fin[f]
                    R = Rs[idx]
                    X = Xs[idx]
                    # find indices of the other two oscillators coupled to current oscillator
                    ind1 = fin[(f + 1)%num_coupled]
                    ind2 = fin[(f + 2)%num_coupled]
                    rs = [self.r[ind1] , self.r[ind2]]
                    phis = [self.PHI[ind1] , self.PHI[ind2]]
                    # state of current oscillator
                    state = [self.PHI[idx], self.r[idx], self.x[idx], self.v[idx], self.m[idx]]
                    t_points = [0,self.dt]
                    solution = scipy.integrate.solve_ivp(
                        fun = lambda t, y: ode_fin(state, omega, R, X, rs, phis),
                        t_span=t_points, 
                        y0=state,
                        method='RK45',
                        t_eval = t_points
                    )
                    try:
                        self.PHI[idx] = solution.y[0, 1]
                        self.r[idx] = solution.y[1, 1]
                        self.x[idx] = solution.y[2, 1]
                        self.v[idx] = solution.y[3, 1]
                        self.m[idx] = solution.y[4, 1]
                    except:
                        print("failed to solve ode with the following: \n")
                        print(f"state= {state}\n")
                        print(f"dt= {self.dt}\n")
                        pass
                    # grab output of oscillator i
                    self.theta[idx] = self.x[idx] + self.r[idx] * np.cos(self.PHI[idx])
                    action[idx] = self.theta[idx]

            # for every back fin
            for fin in back_fins:
                num_coupled = 2
                for f in range(len(fin)):
                    idx = fin[f]                            # grab index of current oscillator
                    ind = fin[(f + 1) % num_coupled]        # grab index of other oscillator
                    rs = [self.r[ind]]
                    phis = [self.PHI[ind]]
                    # state of current oscillator
                    state = [self.PHI[idx], self.r[idx], self.x[idx], self.v[idx], self.m[idx]]
                    t_points = [0,self.dt]
                    solution = scipy.integrate.solve_ivp(
                        fun = lambda t, y: ode_fin(state, omega, R, X, rs, phis),
                        t_span=t_points, 
                        y0=state,
                        method='RK45',
                        t_eval = t_points
                    )
                    try:
                        self.PHI[idx] = solution.y[0, 1]
                        self.r[idx] = solution.y[1, 1]
                        self.x[idx] = solution.y[2, 1]
                        self.v[idx] = solution.y[3, 1]
                        self.m[idx] = solution.y[4, 1]
                    except:
                        print("failed to solve ode with the following: \n")
                        print(f"state= {state}\n")
                        print(f"dt= {self.dt}\n")
                        pass
                    # grab output of oscillator i
                    self.theta[idx] = self.x[idx] + self.r[idx] * np.cos(self.PHI[idx])
                    action[idx] = self.theta[idx]
            
            # record action for time step into total_actions struct
            total_actions[:, i] = action

        return total_actions
    
    def get_other_rollout(self, episode_length=60):
        """
        Calculate the entire rollout. This has all 10 motors coupled together
        Inspired by Auke Isjpeert's 2007 Model of Salamander
        """
        oscillator = {1: [2, 4], 
                      2: [1, 3, 5],
                      3: [2, 6, 8],
                      4: [1, 5],
                      5: [2, 4, 6],
                      6: [3, 5, 10],
                      7: [8, 9],
                      8: [3, 7, 10],
                      9: [7, 10],
                      10: [6, 8, 9]}
        def ode_fin(state, omega, R, X, rs, phis):
            PHI, r, x, v, m= state
            dPhidt = omega
            for i in range(len(rs)):
                r_other = rs[i]
                phi_other = phis[i]
                dPhidt += self.w*r_other*np.sin(phi_other - PHI - self.phi)
            dRdt = v
            dVdt = self.a_r * ((self.a_r/4) * (R-r) - dRdt)
            dXdt = m
            dMdt = self.a_x * ((self.a_x/4) * (X-x) - dXdt)
            return [dPhidt, dRdt, dXdt, dVdt, dMdt]
        # calculate y_out, which in this case we are using as the tau we pass into the turtle
        total_actions = np.zeros((self.num_mods, episode_length))
        omega = self.params[0]
        Rs = self.params[1:self.num_mods + 1]
        Xs = self.params[self.num_mods + 1:]
        for e in range(episode_length):
            action = np.zeros((self.num_mods))
            for i in range(self.num_mods):
                # ex 1: [2, 4], 
                #    2: [1, 3, 5]
                R = Rs[i]
                X = Xs[i]
                fin = oscillator[i + 1]         # for one-indexing
                num_coupled = len(fin)
                fin = [f - 1 for f in fin]
                rs = []
                phis = []
                for c in range(num_coupled):
                    idx = fin[c]
                    rs.append(self.r[idx])
                    phis.append(self.PHI[idx])
                
                state = [self.PHI[i], self.r[i], self.x[i], self.v[i], self.m[i]]
                t_points = [0,self.dt]
                solution = scipy.integrate.solve_ivp(
                    fun = lambda t, y: ode_fin(state, omega, R, X, rs, phis),
                    t_span=t_points, 
                    y0=state,
                    method='RK45',
                    t_eval = t_points
                )
                try:
                    self.PHI[i] = solution.y[0, 1]
                    self.r[i] = solution.y[1, 1]
                    self.x[i] = solution.y[2, 1]
                    self.v[i] = solution.y[3, 1]
                    self.m[i] = solution.y[4, 1]
                except:
                    print("failed to solve ode with the following: \n")
                    print(f"state= {state}\n")
                    print(f"dt= {self.dt}\n")
                    pass
                # grab output of oscillator i
                self.theta[i] = self.x[i] + self.r[i] * np.cos(self.PHI[i])
                action[i] = self.theta[i]

            # record action for time step into total_actions struct
            total_actions[:, e] = action

        return total_actions

    def run(self, 
            env,
            max_episode_length=60):
        """Run an episode.

        Args:
            env: The env object we need to run a step
            max_episode_length: The maximum time window we will allow for
                interactions within a single episode (i.e 2 seconds, 3 seconds, etc.)
                Default is 2 seconds because a typical turtle gait lasts for about that long.
        Returns:
            A tuple (cumulative_reward, total_episode_time).
        """

        # TODO: look into whether you need to normalize the observation or not
        cumulative_reward = 0.0
        observation, __ = env.reset()
        t = 0
        first_time = True
        total_actions = np.zeros((self.num_mods, 1))
        while True:
            dt = 0.05
            action = self.get_action(dt)
            # print(f"action shape: {action.shape}")
            total_actions = np.append(total_actions, action.reshape((6,1)), axis=1)
            # print(f"params: {self.params}\n")
            # print(f"action: {action}\n")
            
            observation, reward, terminated, truncated, info = env.step(action)
            done = truncated or terminated
            cumulative_reward += reward
            # print(info)
            # t = time.time()
            # print(f"reward and cost: {(info['reward_run'], info['reward_ctrl'])}")
            t += 1
            if t > max_episode_length:
                # print(f"we're donesies with {t}")
                break
            if done:
                if truncated:
                    print("truncccc")
                if terminated:
                    print("terminator")
                break
        return cumulative_reward, total_actions
    def set_params_and_run(self,
                           env,
                           policy_parameters,
                           max_episode_length=60,
                           ):
        """Set the the parameters of the policy by copying them
        from the given parameter vector, then run an episode.

        Args:
            policy_parameters: The policy parameters to be used.
            decrease_rewards_by: The reward at each timestep will be
                decreased by this given amount.
            max_episode_length: The maximum number of interactions
                allowed in an episode.
        Returns:
            A tuple (cumulative_reward, t (len of episode in seconds)).
        """
        self.set_parameters(policy_parameters)
        
        cumulative_reward, total_actions = self.run(env,
            max_episode_length=max_episode_length
        )
        return cumulative_reward, total_actions

# def main(args=None):
#     num_params = 21
#     num_mods = 10
#     cpg = AukeCPG(num_params=num_params, num_mods=num_mods, phi=0.0, w=0.5, a_r=20, a_x=20, dt=0.01)
    
#     mu = np.random.rand((num_params)) 
#     sigma = np.random.rand((num_params)) + 0.3

#     ephe = EPHE(
                
#                 # We are looking for solutions whose lengths are equal
#                 # to the number of parameters required by the policy:
#                 solution_length=mu.shape[0],
                
#                 # Population size: the number of trajectories we run with given mu and sigma 
#                 popsize=20,
                
#                 # Initial mean of the search distribution:
#                 center_init=mu,
                
#                 # Initial standard deviation of the search distribution:
#                 stdev_init=sigma,

#                 # dtype is expected as float32 when using the policy objects
#                 dtype='float32', 

#                 K=2
#             )
#     solutions = ephe.ask()     
#     for solution in solutions:
#         print(f"starting params: {solution}\n")
#         eps_len = 800
#         cpg.set_parameters(params=solution)
#         # cpg.reset()
#         total_actions = cpg.get_rollout(episode_length=eps_len)
#         # total_actions = cpg.get_coupled_rollout(episode_length=eps_len)
#         t = np.arange(0, eps_len*cpg.dt, cpg.dt)

#         fig, axs = plt.subplots(nrows=total_actions.shape[0], ncols=1, figsize=(8, 12))
#         for j, ax in enumerate(axs):
#             ax.plot(t, total_actions[j, :])
#             ax.set_title(f"CPG {j+1}")
#             ax.set_xlabel("Time")
#             ax.set_ylabel("Data")
#             ax.grid(True)
#         plt.tight_layout()
#     plt.show()

#     return 0

# if __name__ == "__main__":
#     main()