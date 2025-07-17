import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import rclpy.parameter

from rclpy.parameter_event_handler import ParameterEventHandler

from turtle_interfaces.msg import TurtleTraj, TurtleSensors
import transforms3d.quaternions as quat
from std_msgs.msg import String
from dynamixel_sdk import *                                     # Uses Dynamixel SDK library
from turtle_dynamixel.Dynamixel import *                        # Dynamixel motor class                                  
from turtle_dynamixel.dyn_functions import *                    # Dynamixel support functions
from turtle_dynamixel.torque_control import *                # Controller 
from turtle_dynamixel.Constants import *                        # File of constant variables
from turtle_dynamixel.Mod import *
from turtle_dynamixel.utilities import *
from turtle_dynamixel.utilities import *



class TurtleRobot(Node):

# class TurtleRobot(Node, gym.Env):
    """
    This node is responsible for continously reading sensor data and receiving commands from the keyboard node
    to execute specific trajectoreies or handle emergency stops. It also is responsible for sending motor pos commands to the RL node
    for training and deployment purposes.
    TODO: migrate dynamixel motors into this class
    TLDR; this is the node that handles all turtle hardware things
    """

    def __init__(self, topic, params=None):
        super().__init__('turtle_rl_node')
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=2
        )
        buff_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.declare_parameter('an_int_param', 0)

        self.handler = ParameterEventHandler(self)

        self.callback_handle = self.handler.add_parameter_callback(
            parameter_name="an_int_param",
            node_name="turtle_rl_node",
            callback=self.callback,
        )
        # subscribes to keyboard setting different turtle modes 
        self.mode_cmd_sub = self.create_subscription(
            String,
            topic,
            self.turtle_mode_callback,
            qos_profile)
        # for case when trajectory mode, receives trajectory msg
        self.motor_traj_sub = self.create_subscription(
            TurtleTraj,
            'turtle_traj',
            self.trajectory_callback,
            qos_profile
        )
        # for case camera mode, receives motion primitive
        self.cam_sub = self.create_subscription(
            String,
            'primitive',
            self.primitive_callback,
            buff_profile
        )
        # continously reads from all the sensors and publishes data at end of trajectory
        self.sensors_pub = self.create_publisher(
            TurtleSensors,
            'turtle_sensors',
            qos_profile
        )
        # sends acknowledgement packet to keyboard node
        self.cmd_received_pub = self.create_publisher(
            String,
            'turtle_state',
            qos_profile
        )
        # continously publishes the current motor position      
        self.motor_pos_pub = self.create_publisher(
            String,
            'turtle_motor_pos',
            qos_profile
        )
        self.mode_cmd_sub       # prevent unused variable warning
        self.create_rate(100)
        self.mode = 'rest'      # initialize motors mode to rest state
        self.primitives = ['dwell']
        self.voltage = 12.0
        self.n_axis = 3
        self.qs = np.zeros((10,1))
        self.dqs = np.zeros((10,1))
        self.tvec = np.zeros((1,1))
        if portHandlerJoint.openPort():
            print("[MOTORS STATUS] Suceeded to open port")
        else:
            print("[ERROR] Failed to open port")
        if portHandlerJoint.setBaudRate(BAUDRATE):
            print("[MOTORS STATUS] Suceeded to open port")
        else:
            print("[ERROR] Failed to change baudrate")


        # dynamixel setup
        self.IDs = [1,2,3,4,5,6,7,8,9,10]
        self.nq = 10
        self.Joints = Mod(packetHandlerJoint, portHandlerJoint, self.IDs)
        self.Joints.disable_torque()
        self.Joints.set_current_cntrl_mode()
        self.Joints.enable_torque()
        self.q = np.array(self.Joints.get_position()).reshape(-1,1)
        self.dq = np.array(self.Joints.get_velocity()).reshape(-1,1)
        self.xiao = serial.Serial('/dev/ttyACM0', 115200, timeout=3)   
        self.xiao.reset_input_buffer()
        self.input_history = np.zeros((self.nq,10))
        
        # set thresholds for motor angles 
        self.epsilon = 0.1
        self.min_threshold = np.array([1.60, 3.0, 2.4, 2.43, 1.2, 1.7, 1.45, 1.2, 3.0, 2.3])
        self.max_threshold = np.array([3.45, 5.0, 4.2, 4.5, 4.15, 3.8, 3.2, 4.0, 4.0, 4.7])
        # for PD control
        self.Kp = np.diag([0.8, 0.4, 0.4, 0.8, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4])*2
        self.Kpv = [1] * 6
        self.KD = 0.1
        self.turtle_trajs = ["dive", "straight", "surface", "turnlf", "turnrf"]
        self.voltage_threshold = 11.3

        self.amplitude = np.pi / 180 * np.array([40, 40, 40]).reshape(-1,1)
        self.center = np.pi / 180 * np.array([0, 10, 10])
        self.yaw = 0.8
        self.pitch = 0.5
        self.freq_offset = 0.3
        self.period = 2
        self.ctrl_flag = False
        self.print_sensors = True
        self.end_program = False

        self.q_data = []
        self.dq_data = []
        self.tau_data = []
        self.timestamps = []
        self.qd_data = []
        self.dqd_data = []
        self.ud = [0]*6
        self.td = 0

    def read_params(self, params):
        self.ax_weight = params["ax_weight"]
        self.ay_weight = params["ay_weight"]
        self.az_weight = params["az_weight"]

        self.tau_weight = params["tau_weight"]
        self.quat_weight = params["quat_weight"]


    def callback(self, p: rclpy.parameter.Parameter) -> None:
        self.get_logger().info(f"Received an update to parameter: {p.name}: {rclpy.parameter.parameter_value_to_python(p.value)}")
        self.end_program = True
        self.check_end()
        cmd_msg = String()
        cmd_msg.data = "stop_received"
        self.cmd_received_pub.publish(cmd_msg)
        print("sent stop received msg")
        rclpy.shutdown()

    def turtle_mode_callback(self, msg):
        """
        Callback function that updates the mode the turtle should be in.
        This method is what enables us to set "emergency stops" mid-trajectory. 
        """
        # global mode
        self.mode = msg.data
        # case _:
        match self.mode:
            case "stop":
                self.end_program = True
                self.check_end()
                cmd_msg = String()
                cmd_msg.data = "stop_received"
                self.cmd_received_pub.publish(cmd_msg)
                print("sent stop received msg")
            case "rest":
                self.shutdown_motors()
                if self.print_sensors:
                    print(self.Joints.get_position())
                cmd_msg = String()
                cmd_msg.data = 'rest_received'
                self.cmd_received_pub.publish(cmd_msg)

    def get_turtle_mode(self):
        return self.mode

    def primitive_callback(self, msg):
        """
        Callback function that updates the desired behavior based on the computer vision module. 
        """
        self.primitives.append(msg.data)
        

    def create_data_structs(self):
        """
        Create initial data structs to hold turtle data
        """
        # data structs for recording each rollout
        self.acc_data = np.zeros((self.n_axis,1))
        self.gyr_data = np.zeros((self.n_axis,1))
        self.quat_data = np.zeros((4,1))
        self.tau_data = np.zeros((10,1))
        self.voltage_data = np.zeros((1,1))

        # to store individual contributions of rewards 
        self.a_rewards = np.zeros((1,1))
        self.x_rewards = np.zeros((1,1))
        self.y_rewards = np.zeros((1,1))
        self.z_rewards = np.zeros((1,1))
        self.tau_rewards = np.zeros((1,1))

    def reset(self):
        print("resetting turtle...\n")
        self.Joints.disable_torque()
        self.Joints.enable_torque()
        q = self.Joints.get_position()
        v = self.Joints.get_velocity()
        acc = self.acc_data[:, -1]
        quat = self.quat_data[:, -1]
        observation = np.concatenate((q, v), axis=0)
        return observation, "reset"
    def quatL(self,q):
        qs = q[0]
        if qs == 0:
            print("f QS: {qs}\n\n\n\n\n")
        qv = q[1:]
        o = qs*np.identity(3) + self.skew(qv)
        # qv = qv.reshape((3,1))
        first_row = np.hstack((qs, -qv.T)).reshape(((1,4)))
        second_row = np.hstack((qv.reshape((3,1)), qs*np.identity(3) + self.skew(qv)))
        # return np.array([[qs, -qv.T],[qv, qs*np.identity(3) + self.skew(qv)]])
        return np.vstack((first_row, second_row))

    def inv_q(self,q):
        return np.diag([1, -1, -1, -1]) @ q

    def inv_cayley(self,q):
        print(f"q: {q}")
        return q[1:]/q[0]

    def skew(self,x):
        return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
    
    def _get_reward(self, tau):
        q = self.quat_data[:, -1]
        acc = self.acc_data[:, -1]
        R = self.ay_weight*acc[1]**2 - self.ax_weight*acc[0]**2 - self.az_weight*acc[2]**2 -self.tau_weight* np.linalg.norm(tau)**2 - self.quat_weight * (1 - np.linalg.norm(self.orientation.T @ q))
        return R
    
    def _get_reward_weighted(self, tau):
        reward = 0
        return reward
    
    def step(self, action, PD=False, mode='None'):
        """
        action is tau which we pass into the turtle
        OR its position in radians when PD flag is set to True
        """
        keep_trying = True
        while keep_trying:
            try:
                observation = np.array(self.Joints.get_position())
                v = np.array(self.Joints.get_velocity())
                keep_trying = False
            except:
                print("failed to read from motors")
        if mode == 'SAC':
            qd = np.where(action < self.max_threshold - self.epsilon, action, self.max_threshold - self.epsilon)
            qd = np.where(qd > self.min_threshold + self.epsilon, qd, self.min_threshold + self.epsilon)
            dqd = np.zeros((self.nq,1))
            ddqd = np.zeros((self.nq,1))
            # print(f"dqd shape: {dqd.shape}")
            tau = torque_control(observation.reshape((10,1)),v.reshape((10,1)),qd.reshape((10,1)),dqd,ddqd,self.Kp,self.KD)
            clipped = grab_arm_current(tau, min_torque, max_torque)

            terminated = False
            truncated = False
            info = [v, clipped]
            self.read_sensors()
            reward = self._get_reward(tau)
            obs = self.acc_data[:, -1]
            return [obs, reward, terminated, truncated, info]
        if PD:
            # print(f"action shape: {action.shape}")
            # position control 
            qd = np.where(action < self.max_threshold - self.epsilon, action, self.max_threshold - self.epsilon)
            qd = np.where(qd > self.min_threshold + self.epsilon, qd, self.min_threshold + self.epsilon)
            dqd = np.zeros((self.nq,1))
            ddqd = np.zeros((self.nq,1))
            # print(f"dqd shape: {dqd.shape}")
            tau = torque_control(observation.reshape((10,1)),v.reshape((10,1)),qd.reshape((10,1)),dqd,ddqd,self.Kp,self.KD)
            avg_tau = np.mean(abs(tau))
            clipped = grab_arm_current(tau, min_torque, max_torque)
        else:
            # print(f"action: {action}")
            action = 10e3 * action
            # print(f"action: {action}")

            inputt = grab_arm_current(action, min_torque, max_torque)
            # print(f"observation: {observation}\n")
            clipped = np.where(observation < self.max_threshold - self.epsilon, inputt, np.minimum(np.zeros(len(inputt)), inputt))
            # print(f"clipped1: {clipped}")
            clipped = np.where(observation > self.min_threshold + self.epsilon, clipped, np.maximum(np.zeros(len(inputt)), clipped)).astype('int')
            # print(f"clipped: {clipped}")
            avg_tau = np.mean(abs(clipped))
        # print(f"torque: {clipped}")
        # self.Joints.send_torque_cmd(clipped)
        self.read_sensors()
        reward = self._get_reward(avg_tau)
        terminated = False
        truncated = False
        # return velocity and clipped tau (or radians) passed into motors
        info = [v, clipped]
        return observation, reward, terminated, truncated, info

    def trajectory_callback(self, msg):
        """
        Callback function that takes in list of squeezed arrays
        msg: [qd, dqd, ddqd, tvec]
        """    
        # n = len(msg.tvec)
        # self.qds = np.array(msg.qd).reshape(10,n)
        # self.tvec = np.array(msg.tvec).reshape(1,n)
        # if len(msg.dqd) < 1:
        #     self.mode = 'position_control_mode'
        # else:
        #     self.dqds = np.array(msg.dqd).reshape(10,n)
        #     self.ddqds = np.array(msg.ddqd).reshape(10,n)
        #     self.mode = 'traj_input'
        # self.td
        self.ud = msg.u
        self.td = msg.t

    def teacher_callback(self, msg):
        """
        Has the robot turtle follow the teacher trajectory
        
        """
        n = len(msg.tvec)
        self.qds = np.array(msg.qd).reshape(10,n)
        self.tvec = np.array(msg.tvec).reshape(1,n)
        self.dqds = np.array(msg.dqd).reshape(10,n)
        self.ddqds = np.array(msg.ddqd).reshape(10,n)
        self.mode = 'teacher_traj_input'

    def np2msg(self, mat):
        """
        flattens nd numpy array into a lst for ros pkg messaging
        """
        nq = 10
        squeezed = np.reshape(mat, (nq * mat.shape[1]))
        return squeezed.tolist()
    
    def publish_turtle_data(self):
        """
        At the end of a trajectory (i.e when we set the turtle into a rest state or stop state), turtle node needs to 
        package all the sensor and motor data collected during that trajectory and publish it to the turtle_sensors node
        That way, it can be the local machine and not the turtle machine carrying all the data
        TODO: perhaps look into having turtle save data locally on raspi as well just in case?
        """
        turtle_msg = TurtleSensors()
        
        # # extract and flatten numpy arrays
        # print("quat")
        # quat_x = self.quat_data[0, :]
        # quat_y = self.quat_data[1, :]
        # quat_z = self.quat_data[2, :]
        # quat_w = self.quat_data[3, :]

        # # print("acc")
        # acc_x = self.acc_data[0, :]
        # acc_y = self.acc_data[1, :]
        # acc_z = self.acc_data[2, :]

        # # print("gyr")
        # gyr_x = self.gyr_data[0, :]
        # gyr_y = self.gyr_data[1, :]
        # gyr_z = self.gyr_data[2, :]

        # # print("wuat msg")
        # # # quaternion 
        # # print(f"quat x: {quat_x}\n")
        # quatx = quat_x.tolist()
        # # print(f"quat x: {quatx}\n")
        # # print(f"element type: {type(quatx[0])}")
        # turtle_msg.imu.quat_x = quat_x.tolist()
        # turtle_msg.imu.quat_y = quat_y.tolist()
        # turtle_msg.imu.quat_z = quat_z.tolist()
        # turtle_msg.imu.quat_w = quat_w.tolist()
        # # print("gyr msg")
        # # # angular velocity
        # turtle_msg.imu.gyr_x = gyr_x.tolist()
        # turtle_msg.imu.gyr_y = gyr_y.tolist()
        # turtle_msg.imu.gyr_z = gyr_z.tolist()
        # # print("acc msg")
        # # # linear acceleration
        # turtle_msg.imu.acc_x = acc_x.tolist()
        # turtle_msg.imu.acc_y = acc_y.tolist()
        # turtle_msg.imu.acc_z = acc_z.tolist()
        
        # print("volt msg")
        # print(f"voltage data: {self.voltage_data.tolist()}")
        # # voltage
        # volt_data = self.voltage_data.tolist()
        # print(f"voltage data: {volt_data}")
        # turtle_msg.voltage = volt_data  

        # timestamps and t_0
        # if len(timestamps) > 0:
        #     turtle_msg.timestamps = timestamps.tolist()
        # turtle_msg.t_0 = t_0

        # # motor positions, desired positions and tau data
        # if len(q_data) > 0:
        #     turtle_msg.q = self.np2msg(q_data) 
        # if len(q_data) > 0:
        #     turtle_msg.dq = self.np2msg(dq_data)
        # if len(ddq_data) > 0:
        #     turtle_msg.ddq = self.np2msg(ddq_data)
        # qd_squeezed = self.np2msg(self.qds)
        # turtle_msg.qd = qd_squeezed
        # if len(tau_data) > 0:
        #     turtle_msg.tau = self.np2msg(tau_data)
        # print(self.q_data)
        turtle_msg.q = self.q
        turtle_msg.dq = self.dq
        turtle_msg.timestamps = self.timestamps
        turtle_msg.tau = self.tau_data

        # publish msg 
        self.sensors_pub.publish(turtle_msg)
        print(f"published....")
        # reset the sensor variables for next trajectory recording
        # self.q_data = []
        # self.dq_data = []
        # self.timestamps = []
        # self.tau_data = []

# { "Quat": [ 0.008, 0.015, -0.766, 0.642 ],"Acc": [-264.00, -118.00, 8414.00 ],"Gyr": [-17.00, 10.00, 1.00 ],"Voltage": [ 11.77 ]}
    def read_voltage(self):
        """
        Appends current sensor reading to the turtle node's sensor data structs
        """ 

        self.xiao.reset_input_buffer()
        sensors = self.xiao.readline()
        if self.print_sensors:
            print("testing")
            print(sensors)
        try:
            sensor_dict = json.loads(sensors.decode('utf-8'))
            # print(f"sensor dict: {sensor_dict}")
            sensor_keys = ('Voltage')
            if set(sensor_keys).issubset(sensor_dict):
                volt = sensor_dict['Voltage'][0]
                self.voltage = volt
        except:
            self.voltage = self.voltage
        
        if self.voltage < self.voltage_threshold:
            print("[WARNING] volt reading too low--closing program....")
            self.Joints.disable_torque()
            self.end_program = True
            

    def add_place_holder(self):
        acc = self.acc_data[:, -1].reshape((3,1))
        gyr = self.gyr_data[:, -1].reshape((3,1))
        quat_vec = self.quat_data[:, -1].reshape((4,1))
        self.acc_data = np.append(self.acc_data, acc, axis=1)
        self.gyr_data = np.append(self.gyr_data, gyr, axis=1)
        self.quat_data = np.append(self.quat_data, quat_vec, axis=1)
        self.voltage_data = np.append(self.voltage_data, self.voltage_data[-1])

    def check_end(self):
        if self.end_program:
            self.shutdown_motors()
            return True
        else:
            return False
        
    def shutdown_motors(self):
        # self.Joints.disable_torque()
        # self.Joints.set_current_cntrl_mode()
        # self.Joints.enable_torque()
        self.Joints.send_torque_cmd([0] *len(self.IDs))
        self.Joints.disable_torque()
        # print("Motors Off")
    
    def read_joints(self):
        self.q = self.Joints.get_position()
        self.dq = self.Joints.get_velocity()
    
    def log_joints(self):
        self.q_data.append(self.q)
        self.dq_data.append(self.dq)
    
    def log_time(self, t):
        self.timestamps.append(t)
    
    def log_u(self, u):
        self.tau_data.append(u)

    def log_desired_state(self, qd, dqd):
        self.qd_data = qd
        self.dqd_data = dqd

    def get_state(self):
        return self.q, self.dq

    def get_state_np(self):
        return np.array(self.q).reshape(-1,1), np.array(self.dq).reshape(-1,1)
    
    def read_sensors(self):
        """
        Appends current sensor reading to the turtle node's sensor data structs
        """        
        no_check = False
        keep_trying = True
        attempts = 0
        while keep_trying:
            if attempts >= 3:
                print("adding place holder")
                self.add_place_holder()
                break
            self.xiao.reset_input_buffer()
            sensors = self.xiao.readline()
            # print(sensors)
            # make sure it's a valid byte that's read
            if len(sensors) != 0:
                # this ensures the right json string format
                if sensors[0] == 32 and sensors[-1] == 10:
                    try:
                        sensor_dict = json.loads(sensors.decode('utf-8'))
                    except:
                        no_check = True
                    # add sensor data
                    if no_check == False:
                        sensor_keys = ('Acc', 'Gyr', 'Quat', 'Voltage')
                        if set(sensor_keys).issubset(sensor_dict):
                            a = np.array(sensor_dict['Acc']).reshape((3,1)) * 9.81
                            gyr = np.array(sensor_dict['Gyr']).reshape((3,1))
                            q = np.array(sensor_dict["Quat"])
                            R = quat.quat2mat(q)
                            g_w = np.array([[0], [0], [9.81]])
                            g_local = np.dot(R.T, g_w)
                            acc = a - g_local           # acc without gravity 
                            quat_vec = q.reshape((4,1))
                            volt = sensor_dict['Voltage'][0]
                            self.acc_data = np.append(self.acc_data, acc, axis=1)
                            self.gyr_data = np.append(self.gyr_data, gyr, axis=1)
                            self.quat_data = np.append(self.quat_data, quat_vec, axis=1)
                            self.voltage_data = np.append(self.voltage_data, volt)
                            self.voltage = volt
                            keep_trying = False
            attempts += 1

def turtle_main():
    rclpy.init()
    turtle_node = TurtleRobot('turtle_mode_cmd')
    rclpy.spin(turtle_node)
    rclpy.shutdown()

if __name__ == '__main__':
    turtle_main()