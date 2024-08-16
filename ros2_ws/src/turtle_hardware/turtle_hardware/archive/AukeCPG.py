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
        self.num_params = num_params
        print(f"starting phi: {self.phi}\n")
        print(f"starting w: {self.w}\n")
        print(f"number of CPG modules: {self.num_mods}\n")
    def filter(self, params):
        scaled = params.copy()
        scaled[0] = np.clip(scaled[0], 2.0, 7)
        scaled[1:self.num_mods + 1]= np.clip(scaled[1:self.num_mods + 1], 0.0, 1.0)
        scaled[self.num_mods + 1:]= np.clip(scaled[self.num_mods + 1:], 0, np.pi)
        return scaled
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
        # self.params = params        
        scaled = params.copy()
        scaled[0] = np.clip(scaled[0], 2.0, 10)
        scaled[1:self.num_mods + 1]= np.clip(scaled[1:self.num_mods + 1], 0.0, 1.0)
        scaled[self.num_mods + 1:]= np.clip(scaled[self.num_mods + 1:], 0, np.pi)
        self.params = scaled

        print(f"current params: {self.params}")
    
    def get_params(self):
        return self.params
    
    def reset(self):
        """
        Reset your CPGs?
        """
        self.PHI = np.zeros((self.num_mods))
        self.r = np.zeros((self.num_mods))
        self.x = np.zeros((self.num_mods))
        self.m = np.zeros((self.num_mods))  
        self.v = np.zeros((self.num_mods))  

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
                self.theta[m] = np.pi + (np.pi - self.theta[m])
            # self.theta[m] += np.pi
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
            # print(f"time step: {i}")
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
                if m in [3, 4, 5, 8, 9]:
                    print(self.x[m])
                #     self.r[m] = 1 * self.r[m]
                self.theta[m] = self.x[m] + self.r[m] * np.cos(self.PHI[m])
                # grab output of oscillator i
                # if m in [3, 4, 5, 8, 9]:
                #     self.theta[m] = -1 * self.theta[m]
                # 4/30/24 comment: during the last pool test we had this line uncommented-- this could explain some things
                # self.theta[m] += np.pi/2
                action[m] = self.theta[m]            
            # record action for time step into total_actions struct
            total_actions[:, i] = action

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
            action = self.get_action()
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