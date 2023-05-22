class Action:
    def __init__(self, t_power, t_recovery, F_catch, F_finish, tau_catch, tau_finish):
        self.t_power    = t_power
        self.t_recovery = t_recovery
        self.F_catch    = F_catch
        self.F_finish   = F_finish
        self.tau_catch  = tau_catch
        self.tau_finish = tau_finish

class Primitive:
    def __init__(self, name, action):
        self.action = action
        self.name = name
        