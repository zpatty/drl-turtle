import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import rclpy.parameter

from rclpy.parameter_event_handler import ParameterEventHandler

import os, sys
submodule = os.path.expanduser("~") + "/drl-turtle/ros2_ws/install/turtle_hardware/lib/python3.12/site-packages/turtle_hardware/"
sys.path.append(submodule)
import transforms3d.quaternions as quat
from turtle_interfaces.msg import TurtleTraj, TurtleSensors, TurtleCtrl, TurtleMode
from std_msgs.msg import String, Float32MultiArray
from dynamixel_sdk import *                                     # Uses Dynamixel SDK library
from turtle_dynamixel.Dynamixel import *                        # Dynamixel motor class                                  
from turtle_dynamixel.dyn_functions import *                    # Dynamixel support functions
from turtle_dynamixel.Constants import *                        # File of constant variables
from turtle_dynamixel.Mod import *
from turtle_dynamixel.utilities import *
from turtle_dynamixel.utilities import *
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup




class TurtleRobot(Node):

# class TurtleRobot(Node, gym.Env):
    """
    This node is responsible for continously reading sensor data and receiving commands from the keyboard node
    to execute specific trajectoreies or handle emergency stops. It also is responsible for sending motor pos commands to the RL node
    for training and deployment purposes.
    TODO: migrate dynamixel motors into this class
    TLDR; this is the node that handles all turtle hardware things
    """

    def __init__(self, params=None):
        super().__init__('turtle_hardware_node')
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
        timer_cb_group = None
        self.call_timer = self.create_timer(0.02, self._timer_cb, callback_group=timer_cb_group)

        self.watchdog_timer = self.create_timer(0.04, self._watchdog_cb, callback_group=timer_cb_group)


        # # subscribes to keyboard setting different turtle modes 
        self.mode_sub = self.create_subscription(
            TurtleMode,
            'turtle_mode',
            self.turtle_mode_callback,
            qos_profile)
        
        # # subscribes to keyboard setting different turtle modes 
        self.u_sub = self.create_subscription(
            TurtleTraj,
            'turtle_u',
            self.turtle_input_callback,
            qos_profile)

        # continously reads from all the sensors and publishes data at end of trajectory
        self.sensors_pub = self.create_publisher(
            TurtleSensors,
            'turtle_sensors',
            qos_profile
        )

        self.create_rate(100)
        self.mode = 'rest'      # initialize motors mode to rest state
        self.primitives = ['dwell']
        self.voltage = 12.0



        self.t0 = time.time()
        # dynamixel setup
        self.IDs = [1,2,3,4,5,6,7,8,9,10]
        self.nq = 10

        if portHandlerJoint.openPort():
            print("[MOTORS STATUS] Suceeded to open port")
        else:
            print("[ERROR] Failed to open port")
        if portHandlerJoint.setBaudRate(BAUDRATE):
            print("[MOTORS STATUS] Suceeded to open port")
        else:
            print("[ERROR] Failed to change baudrate")

        self.Joints = Mod(packetHandlerJoint, portHandlerJoint, self.IDs)
        self.Joints.disable_torque()
        self.Joints.set_current_cntrl_mode()
        self.Joints.enable_torque()

        # # sensors and actuators
        self.q = np.array(self.Joints.get_position()).reshape(-1,1)
        self.dq = np.array(self.Joints.get_velocity()).reshape(-1,1)

        # self.q = np.zeros((self.nq,1))
        # self.dq = np.zeros((self.nq,1))
        self.qd = np.zeros((self.nq,1))
        self.dqd = np.zeros((self.nq,1))
        self.u = [0]*self.nq
        self.input = [0]*self.nq
        self.t = 0
        self.acc = np.zeros((3,1))
        self.gyr = np.zeros((3,1))
        self.quat_vec = np.zeros((4,1))


        self.xiao = serial.Serial('/dev/ttyACM0', 115200, timeout=3)   
        self.xiao.reset_input_buffer()
        self.input_history = np.zeros((self.nq,10))
        
        self.voltage_threshold = 11.3

        self.print_sensors = True

        # arrays to store trajectories if desired
        self.q_data = []
        self.dq_data = []
        self.u_data = []
        self.input_data = []
        self.timestamps = []
        self.qd_data = []
        self.dqd_data = []

        self.food = True
        self.first_bark = False

        # t = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        # self.folder_name =  "data/" + t
        # os.makedirs(self.folder_name)
        


    def _timer_cb(self):
        
        self.read_joints()
        self.read_sensors()
        if self.voltage < self.voltage_threshold:
            self.get_logger().info('Battery Low, Please Charge')
            self.mode = "rest"
        self.t = time.time() - self.t0
        if self.mode == 'p':
            
            self.input = grab_arm_current(self.u, min_torque, max_torque)
            # self.Joints.send_torque_cmd(self.input)
        elif self.mode == 'v':
            self.input = np.clip(self.u * 30.0 / np.pi / 0.229, -160., 160.).astype(int).squeeze().tolist()
            # self.Joints.send_vel_cmd(self.input)
        self.publish_turtle_data()
        # self.get_logger().info('Received response')
        self.log_data()

    def turtle_input_callback(self, msg):
        self.u = self.convert_u_to_motors(np.array(msg.u))
        self.qd = np.array(msg.qd)
        self.dqd = np.array(msg.dqd)
        self.food = True

    def _watchdog_cb(self):
        if not self.food:
            self.get_logger().info('Bark Bark!')
            self.mode = "rest"
            self.shutdown_motors()
        self.food = False
        self.first_bark = False

    def turtle_mode_callback(self, msg):
        self.mode = msg.mode
        print(self.mode)
        match self.mode:
            case "rest":
                self.shutdown_motors()
                # cmd_msg = String()
                # cmd_msg.data = 'rest mode'
                # self.cmd_received_pub.publish(cmd_msg)
                print(self.q)
                self.get_logger().info('Rest Mode!')
            case "v":
                self.Joints.disable_torque()
                self.Joints.set_velocity_cntrl_mode()
                self.Joints.enable_torque()
                self.get_logger().info('Velocity Mode!')
                # cmd_msg = String()
                # cmd_msg.data = 'velocity ctrl'
                # self.cmd_received_pub.publish(cmd_msg)
            case "p":
                self.Joints.disable_torque()
                self.Joints.set_current_cntrl_mode()
                self.Joints.enable_torque()
                self.get_logger().info('Position Mode!')
                # cmd_msg = String()
                # cmd_msg.data = 'compliant position ctrl'
                # self.cmd_received_pub.publish(cmd_msg)

    def get_turtle_mode(self):
        return self.mode

    def np2msg(self, mat):
        """
        flattens nd numpy array into a lst for ros pkg messaging
        """
        nq = 10
        squeezed = np.reshape(mat, (nq * mat.shape[1]))
        return squeezed.tolist()
    
    def publish_turtle_data(self):
        """
        Send data out
        """
        turtle_msg = TurtleSensors()
        # turtle_msg.q = self.q
        # turtle_msg.dq = self.dq
        turtle_msg.q, turtle_msg.dq = self.convert_motors_to_q(self.q, self.dq)
        turtle_msg.t = self.t
        _, turtle_msg.u = self.convert_motors_to_q(self.u, self.u)
        turtle_msg.qd = self.qd
        turtle_msg.dqd = self.dqd
        turtle_msg.input = self.input
        turtle_msg.imu.quat = self.quat_vec
        # # angular velocity
        turtle_msg.imu.gyr = self.gyr
        # print("acc msg")
        # # linear acceleration
        turtle_msg.imu.acc = self.acc

        # publish msg 
        self.sensors_pub.publish(turtle_msg)

    def convert_motors_to_q(self, q, dq):
        q = np.array(q).reshape(10,)  - np.pi
        qd = np.array(dq).reshape(10,)
        q_new = np.hstack((-q[3:6], [q[0], -q[1], -q[2]], q[6:]))
        qd_new = np.hstack((-qd[3:6], [qd[0], -qd[1], -qd[2]], qd[6:]))
        return q_new, qd_new

    def convert_u_to_motors(self, u):
        if np.size(u) == 6:
            u = np.hstack((u, np.zeros(4,)))
        u_new = np.hstack(([u[3], -u[4], -u[5]], -u[0:3], u[6:]))
        return u_new.reshape(-1,1)


        
    def shutdown_motors(self):
        if self.mode == 'p':
            self.Joints.send_torque_cmd([0] *len(self.IDs))
        elif self.mode == 'v':
            self.Joints.send_vel_cmd([0] *len(self.IDs))
        self.Joints.disable_torque()
        # print("Motors Off")
        pass
    
    def read_joints(self):
        self.q = self.Joints.get_position()
        self.dq = self.Joints.get_velocity()
    
    def log_joints(self):
        self.q_data.append(self.q)
        self.dq_data.append(self.dq)
    
    def log_time(self, t):
        self.timestamps = t
    
    def log_u(self, u):
        self.tau_data = u
    
    def log_data(self):
        self.q_data.append(self.q)
        self.dq_data.append(self.dq)
        self.u_data.append(self.q)
        self.dq_data.append(self.dq)
        self.input_data.append(self.dq)
        self.timestamps.append(self.t)
        self.qd_data.append(self.qd)
        self.dqd_data.append(self.dqd)

    def save_data(self):
        np.savez(self.folder_name + "_np_data", q=self.q_data, dq=self.dq_data, t=self.timestamps, u=self.u_data, input=self.input_data, qd=self.qd_data, dqd=self.dqd_data)  

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
            if attempts >= 1:
                # print("adding place holder")
                # self.add_place_holder()
                break
            self.xiao.reset_input_buffer()
            sensors = self.xiao.readline()
            # self.get_logger().info(sensors)
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
                            self.gyr = np.array(sensor_dict['Gyr']).reshape((3,1))
                            q = np.array(sensor_dict["Quat"])
                            R = quat.quat2mat(q)
                            g_w = np.array([[0], [0], [9.81]])
                            g_local = np.dot(R.T, g_w)
                            self.acc = a - g_local           # acc without gravity 
                            self.quat_vec = q.reshape((4,1))
                            volt = sensor_dict['Voltage'][0]
                            # self.acc_data = np.append(self.acc_data, acc, axis=1)
                            # self.gyr_data = np.append(self.gyr_data, gyr, axis=1)
                            # self.quat_data = np.append(self.quat_data, quat_vec, axis=1)
                            # self.voltage_data = np.append(self.voltage_data, volt)
                            self.voltage = volt
                            keep_trying = False
            attempts += 1

def main():
    rclpy.init()
    turtle_node = TurtleRobot()
    try:
        rclpy.spin(turtle_node)
        # rclpy.shutdown()
    except KeyboardInterrupt:
        turtle_node.shutdown_motors()
        print("shutdown")
    except Exception as e:
        turtle_node.shutdown_motors()
        print("some error occurred")
        # turtle_node.shutdown_motors()
        exec_type, obj, tb = sys.exc_info()
        fname = os.path.split(tb.tb_frame.f_code.co_filename)[1]
        print(exec_type, fname, tb.tb_lineno)
        print(e)
    # turtle_node.save_data()
    

if __name__ == '__main__':
    main()