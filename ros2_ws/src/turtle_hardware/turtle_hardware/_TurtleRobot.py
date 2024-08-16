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
from turtle_dynamixel.turtle_controller import *                # Controller 
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

    def __init__(self, topic, params=None):
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
        self.call_timer = self.create_timer(1, self._timer_cb, callback_group=timer_cb_group)


        self.declare_parameter('mode', 'rest')

        self.handler = ParameterEventHandler(self)

        self.callback_handle = self.handler.add_parameter_callback(
            parameter_name="mode",
            node_name="turtle_hardware_node",
            callback=self.callback,
        )
        # subscribes to keyboard setting different turtle modes 
        self.mode_cmd_sub = self.create_subscription(
            String,
            topic,
            self.turtle_mode_callback,
            qos_profile)
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

        self.t0 = time.time()
        # dynamixel setup
        self.IDs = [1,2,3,4,5,6,7,8,9,10]
        self.nq = 10
        self.Joints = Mod(packetHandlerJoint, portHandlerJoint, self.IDs)
        self.Joints.disable_torque()
        self.Joints.set_current_cntrl_mode()
        self.Joints.enable_torque()

        # sensors and actuators
        self.q = np.array(self.Joints.get_position()).reshape(-1,1)
        self.dq = np.array(self.Joints.get_velocity()).reshape(-1,1)
        self.u = [0]*10
        self.t = 0
        self.acc = np.zeros((3,1))
        self.gyr = np.zeros((3,1))
        self.quat_vec = np.zeros((4,1))



        self.xiao = serial.Serial('/dev/ttyACM0', 115200, timeout=3)   
        self.xiao.reset_input_buffer()
        self.input_history = np.zeros((self.nq,10))
        
        # set thresholds for motor angles 
        self.epsilon = 0.1
        self.min_threshold = np.array([1.60, 3.0, 2.4, 2.43, 1.2, 1.7, 1.45, 1.2, 3.0, 2.3])
        self.max_threshold = np.array([3.45, 5.0, 4.2, 4.5, 4.15, 3.8, 3.2, 4.0, 4.0, 4.7])
        self.voltage_threshold = 11.3

        self.amplitude = np.pi / 180 * np.array([40, 40, 70]).reshape(-1,1)
        self.center = np.pi / 180 * np.array([-30, 10, -10])
        self.yaw = 0.5
        self.pitch = 0.5
        self.freq_offset = 0.5
        self.period = 2
        self.ctrl_flag = False
        self.print_sensors = True
        self.end_program = False

        # arrays to store trajectories if desired
        self.q_data = []
        self.dq_data = []
        self.tau_data = []
        self.timestamps = []
        self.qd_data = []
        self.dqd_data = []
        


    def _timer_cb(self):
        self.get_logger().info('Timer')
        self.read_joints()
        self.read_sensors()
        if self.voltage < self.voltage_threshold:
            self.get_logger().info('Battery Low, Please Charge')
            self.mode = "rest"
        self.t = time.time() - self.t0
        if self.mode == 'p':
            curr = grab_arm_current(self.u, min_torque, max_torque)
            self.Joints.send_torque_cmd(curr)
        elif self.mode == 'v':
            vd_ticks = np.clip(self.u * 30.0 / np.pi / 0.229, -160., 160.).astype(int).squeeze().tolist()
            self.Joints.send_vel_cmd(vd_ticks)
        self.publish_turtle_data()
        # self.get_logger().info('Received response')


    def callback(self, p: rclpy.parameter.Parameter) -> None:
        self.mode = rclpy.parameter.parameter_value_to_python(p.value)
        self.get_logger().info(f"Received an update to parameter: {p.name}: {self.mode}")
        match self.mode:
            case "rest":
                self.shutdown_motors()
                cmd_msg = String()
                cmd_msg.data = 'rest mode'
                self.cmd_received_pub.publish(cmd_msg)
            case "v":
                self.Joints.disable_torque()
                self.Joints.set_velocity_cntrl_mode()
                self.Joints.enable_torque()
                cmd_msg = String()
                cmd_msg.data = 'velocity ctrl'
                self.cmd_received_pub.publish(cmd_msg)
            case "p":
                self.Joints.disable_torque()
                self.Joints.set_current_cntrl_mode()
                self.Joints.enable_torque()
                cmd_msg = String()
                cmd_msg.data = 'compliant position ctrl'
                self.cmd_received_pub.publish(cmd_msg)

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
        # print(turtle_msg)
        turtle_msg.q = self.q
        turtle_msg.dq = self.dq
        turtle_msg.timestamps = self.t
        turtle_msg.tau = self.u

        # publish msg 
        self.sensors_pub.publish(turtle_msg)

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
        


    def check_end(self):
        if self.end_program:
            self.shutdown_motors()
            return True
        else:
            return False
        
    def shutdown_motors(self):

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
        self.timestamps = t
    
    def log_u(self, u):
        self.tau_data = u

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
            print(sensors)
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
                            if self.voltage < self.voltage_threshold:
                                print("[WARNING] volt reading too low--closing program....")
                                self.shutdown_motors()
                            keep_trying = False
            attempts += 1

def turtle_main():
    rclpy.init()
    turtle_node = TurtleRobot('turtle_mode_cmd')
    try:
        rclpy.spin(turtle_node)
        rclpy.shutdown()
    except Exception as e:
        turtle_node.shutdown_motors()
        print("some error occurred")
        # turtle_node.shutdown_motors()
        exec_type, obj, tb = sys.exc_info()
        fname = os.path.split(tb.tb_frame.f_code.co_filename)[1]
        print(exec_type, fname, tb.tb_lineno)
        print(e)
    

if __name__ == '__main__':
    turtle_main()