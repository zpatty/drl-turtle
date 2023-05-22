import numpy as np

# Default Robot Params
default_m = 5
robot_height = 0.2
robot_length = 1
robot_width = 0.5
# square everything
robot_height2 = robot_height**2
robot_length2 = robot_length**2
robot_width2 = robot_width**2

drag_coefficient = 0.9 # (estimate from a box)
front_area = robot_height*robot_width
water_density = 1024 # [kg/m3]

recovery_stroke_penalty = 1.1 # 10% more drag on recovery

default_power_drag_coef = drag_coefficient*water_density*front_area/2
default_recovery_drag_coef = default_power_drag_coef*recovery_stroke_penalty

water_viscosity = 1.02*10**-3  # 1.02mPas
# assume cylinder in all directions (doesn't make sense but whatever)
average_distance = (robot_width + robot_height + robot_length)/3
average_area = np.pi*average_distance*average_distance

angular_drag_factor = 1000 # entirely made up to make drag higher 
default_angular_drag_coef = average_area*water_viscosity*angular_drag_factor



default_j = np.diag(np.array([robot_height2+robot_length2, robot_width2+robot_length2, robot_height2+robot_width2])*default_m/12)


class Robot:
    def __init__(self, m=default_m, J=default_j, power_drag_coef=default_power_drag_coef, recovery_drag_coef=default_recovery_drag_coef, angular_drag_coef=default_angular_drag_coef):
        self.m = m
        self.J = J
        self.J_inv = np.linalg.inv(J)
        self.power_drag_coef = power_drag_coef
        self.recovery_drag_coef = recovery_drag_coef
        self.angular_drag_coef = angular_drag_coef
