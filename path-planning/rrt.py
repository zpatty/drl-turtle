import numpy as np
import random
import matplotlib.pyplot as plt
from Primitive import Action
from State import State
from Environment import Environment
from ShortList import ShortList
from Robot import Robot


class RRT:
    def __init__(self, environment : Environment, initial_state : State, target_state : State, robot : Robot, max_target_cost=1):
        self.environment = environment
        self.initial_state = initial_state
        self.target_state = target_state
        self.robot = robot
        self.max_target_cost = max_target_cost

        self.tree = {initial_state : None}
        self.child_tree = {initial_state : []}

        self.tolerance = 1.5

        self.position_weight = 1
        self.orientation_weight = 0
        self.linear_velocity_weight = 0
        self.angular_velocity_weight = 0


        # variables that weight going towards the target state by some amount 
        self.target_position_weight = 0.1
        self.target_orientation_weight = 0  
        self.target_linear_velocity_weight = 0 
        self.target_angular_velocity_weight = 0

        self.attempted_actions = 6 # sample 6 actions in each "extend" call


    def get_random_action(self):
        t_recovery = random.random()*0.5 + 1 # range (1,1.5)
        t_power= random.random()*0.25 + 0.5 # range (0.5,0.75)
        
        F_catch  = np.random.rand(3)*np.array([100,0,0]) 
        F_finish = np.random.rand(3)*np.array([100,0,0]) 
        tau_catch = np.random.rand(3)*np.array([0,0,3]) - np.array([0,0,1.5])
        tau_finish = np.random.rand(3)*np.array([0,0,3]) - np.array([0,0,1.5])

        return Action(t_power, t_recovery, F_catch, F_finish, tau_catch, tau_finish)


    def cost(self, state : State, other_state : State):
        position_cost = np.linalg.norm(state.r - other_state.r)
        return position_cost
    
    
    def get_random_state(self):
        if random.random() < 0.3:
            return self.target_state
        
        new_r = np.random.rand(3)
        new_r *= self.environment.high_location - self.environment.low_location
        new_r += self.environment.low_location

        new_q = np.array([0,0,0,0]).T
        new_v = np.array([0,0,0]).T
        new_w = np.array([0,0,0]).T
        

        return State(new_r,new_q,new_v,new_w)

        
    def add_point(self):
        new_state = self.get_random_state()

        short_list = ShortList(5)

        # Get top n closest states to the new state
        for state in self.tree.keys():
            cost = self.cost(state, new_state)
            short_list.insert(state,cost)

        best_cost = float("inf")
        best_state = None
        best_action = None
        best_parent_state = short_list.front()[0]

        # Get the action which puts the next state closest to target state
        for _ in range(self.attempted_actions):
            action = self.get_random_action()

            next_state = best_parent_state.step_new(action, self.robot, environment, self.tolerance)

            if next_state is None:
                continue

            cost = self.cost(next_state, new_state)
            if cost < best_cost:
                best_state = next_state
                best_action = action
                best_cost = cost
    
        if best_state is None:
            return None

        self.tree[best_state] = (best_parent_state, best_action)
        
        self.child_tree[best_parent_state].append(best_action)
        self.child_tree[best_state] = []

        return best_state
    
    def run(self):
        num_points = 0
        num_iterations = 0
        cost = float("inf")
        while (cost > self.max_target_cost):
            if num_iterations > 3000:
                print("Too many iterations")
                return None
            num_iterations += 1
            new_state = self.add_point()
            if new_state is None:
                continue
            cost = self.cost(new_state, self.target_state)
            num_points+=1
            if not num_points%25:
                print(f"Added {num_points} points")
            if num_points > 200:
                print("DID NOT CONVERGE")
                return None
        
        path = []
        parent = self.tree[new_state]
        while parent:
            path.append(parent)
            parent = self.tree[parent[0]]
        path.reverse()
        return path

    def add_and_visualise(self,ax):
        best_state = self.add_point()
        parent = self.tree[best_state]
        parent[0].step_new(parent[1], self.robot, ax=ax)


    def visualise_tree(self, path=None):
        fig, ax = plt.subplots()
        ax.set_xlim((self.environment.low_location[0]-1,self.environment.high_location[0]+1))
        ax.set_ylim((self.environment.low_location[0]-1,self.environment.high_location[0]+1))
        
        for state, parent in self.tree.items():
            if parent is None:
                ax.plot(state.r[0], state.r[1], marker="o", markersize=5, markeredgecolor="black", markerfacecolor="green")
            else:
                parent[0].step_new(parent[1], self.robot, self.environment, 0, ax=ax)
        
        target_circle = plt.Circle((self.target_state.r[0], self.target_state.r[1]), self.max_target_cost**0.5, color='r')
        ax.add_patch(target_circle)

        if path is not None:
            for state, primitive in path:
                if primitive is None:
                    ax.plot(state.r[0], state.r[1], marker="o", markersize=5, markeredgecolor="orange", markerfacecolor="green")
                else:
                    state.step_new(primitive, self.robot, self.environment, 0, ax=ax, color="orange")


        for obstruction in self.environment.obstructions.T:
            obstructed_circle = plt.Circle((obstruction[0], obstruction[1]), self.tolerance, color='b')
            ax.add_patch(obstructed_circle)

        plt.show()



if __name__ == "__main__":
    
    robot = Robot()

    # Inital State
    initial_r = np.array([8,8,0])
    initial_q = np.array([0,0,1,0])
    initial_v = np.array([1,0,0])
    initial_w = np.array([0,0,0])

    initial_state = State(initial_r, initial_q, initial_v, initial_w)

    # Target State
    target_r = np.array([2,2,0])
    target_q = np.array([1,0,0,0])
    target_v = np.array([0,0,0])
    target_w = np.array([0,0,0])

    target_state = State(target_r, target_q, target_v, target_w)

    # Environment

    low_boundary = np.array([0,0,-0.5])
    high_boundary = np.array([10,10,0.5])
    obstructions = np.array([[5,5.5,0],[8,3,0], [5,0,0], [2,7,0]]).T

    environment = Environment(low_boundary,high_boundary,obstructions)
    
    import time
    
    rrt = RRT(environment,initial_state, target_state, robot, 1)


    start = time.time()
    path = rrt.run()
    end = time.time()
 
    rrt.visualise_tree(path=path)

    print(path)
