#!/usr/bin/python

# Standard imports
import rclpy
from rclpy.node import Node
from turtle_interfaces.msg import TurtleTraj, TurtleSensors
import transforms3d.quaternions as quat
import torch
from EPHE import EPHE
from DualCPG import DualCPG
from AukeCPG import AukeCPG
from std_msgs.msg import String
from dynamixel_sdk import *                                     # Uses Dynamixel SDK library
from turtle_dynamixel.Dynamixel import *                        # Dynamixel motor class                                  
from turtle_dynamixel.dyn_functions import *                    # Dynamixel support functions
from turtle_dynamixel.turtle_controller import *                # Controller 
from turtle_dynamixel.Constants import *                        # File of constant variables
from turtle_dynamixel.Mod import *
from turtle_dynamixel.utilities import *
from turtle_dynamixel.utilities import *
from TurtleRobot import *                                       # Turtle node class
import numpy as np
import json
import serial
import random
import cv2

# for dynamixel motors 
os.system('sudo /home/crush/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/turtle_dynamixel/latency_write.sh')

# global variable was set in the callback function directly
global mode
mode = 'rest'
def parse_cv_params():
    with open('cv_config.json') as config:
        param = json.load(config)
    print(f"[MESSAGE] Config: {param}\n")    
    # Serializing json
    config_params = json.dumps(param, indent=14)
    
    return param, config_params
def parse_learning_params():
    with open('turtle_config/config.json') as config:
        param = json.load(config)
    print(f"[MESSAGE] Config: {param}\n")    
    # Serializing json
    config_params = json.dumps(param, indent=14)
    
    return param, config_params
def main(args=None):
    rclpy.init(args=args)
    threshold = 11.3
    # params, config_params = parse_learning_params()
    turtle_node = TurtleRobot('turtle_mode_cmd', params=params)
    q = np.array(turtle_node.Joints.get_position()).reshape(-1,1)
    print(f"Our initial q: " + str(q))
    # create folders 
    # try:
    #     typed_name =  input("give folder a name: ")
    #     folder_name = "data/" + typed_name
    #     os.makedirs(folder_name, exist_ok=False)

    # except:
    #     print("received weird input or folder already exists-- naming folder with timestamp")
    #     t = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    #     folder_name =  "data/" + t
    #     os.makedirs(folder_name)
    # create_shell_script(params=params, folder_name=folder_name)

    # best_param_fname = folder_name + f'/turtle_node.best_params.pth'
    # config_fname = folder_name + "/config.json"
    # create shell script for sending data over
    

    config_params = json.dumps(params, indent=10)

    # save config file to specified folder beforehand
    # with open(config_fname, 'w') as outfile:
    #     outfile.write(config_params)

    best_reward = 0
    try: 
        while rclpy.ok():
            rclpy.spin_once(turtle_node)
            print(f"turtle mode: {turtle_node.mode}\n")
            if turtle_node.voltage < threshold:
                print("[WARNING] volt reading too low--closing program....")
                turtle_node.Joints.disable_torque()
                break
            if turtle_node.mode == 'stop':
                print("ending entire program...")
                print("disabling torques entirely...")
                turtle_node.Joints.send_torque_cmd([0] *len(turtle_node.IDs))
                turtle_node.Joints.disable_torque()
                cmd_msg = String()
                cmd_msg.data = "stop_received"
                turtle_node.cmd_received_pub.publish(cmd_msg)
                print("sent stop received msg")
                break
            elif turtle_node.mode == 'rest':
                rclpy.spin_once(turtle_node)
                turtle_node.read_voltage()
                if turtle_node.voltage < threshold:
                    turtle_node.Joints.disable_torque()
                    print(f"VOLTAGE: {turtle_node.voltage}")
                    print("THRESHOLD MET TURN OFFFF")
                    break
                # print(turtle_node.Joints.get_position())
                # time.sleep(0.5)
                # turtle_node.Joints.send_torque_cmd([0] *len(turtle_node.IDs))
                turtle_node.Joints.disable_torque()
                print(turtle_node.Joints.get_position())
                cmd_msg = String()
                cmd_msg.data = 'rest_received'
                turtle_node.cmd_received_pub.publish(cmd_msg)
                print(f"current voltage: {turtle_node.voltage}\n")


            elif turtle_node.mode == 'cv':
                """
                Test cv stuff
                """

                cv_params, __ = parse_cv_params()

                DIM=(1920, 1080)
                KL=np.array([[914.6609693549937, 0.0, 996.710617938969], [0.0, 967.9244752752224, 531.9164424060089], [0.0, 0.0, 1.0]])
                DL=np.array([[-0.1356783973167512], [0.15271796879021393], [-0.14927909026390898], [0.054553322922445247]])
                KR=np.array([[894.3158759020713, 0.0, 1005.5147253984019], [0.0, 953.7162638446257, 550.0046766951555], [0.0, 0.0, 1.0]])
                DR=np.array([[-0.03029069271100218], [-0.05098557630346465], [0.03042968864943995], [-0.007140226075471247]])
                R=np.array([[0.8778242267055131, 0.03825565357540778, -0.4774527536609107], [-0.017035265337028843, 0.9986682915118547, 0.04869746670711228], [0.47867987919251936, -0.03461428171017962, 0.8773069159410083]])
                T=np.array([[-3.0558948932592864], [0.09397400596710861], [-0.8536105947709979]])

                R1,R2,P1,P2,Q = cv2.fisheye.stereoRectify(KL,DL,KR,DR,DIM,R,T, cv2.fisheye.CALIB_ZERO_DISPARITY)

                L_undist_map=cv2.fisheye.initUndistortRectifyMap(KL,DL,np.identity(3),KL,DIM,cv2.CV_32FC1)
                R_undist_map=cv2.fisheye.initUndistortRectifyMap(KR,DR,np.identity(3),KR,DIM,cv2.CV_32FC1)

                left1, left2 = cv2.fisheye.initUndistortRectifyMap(KL, DL, R1, P1, DIM, cv2.CV_32FC1)
                right1, right2 = cv2.fisheye.initUndistortRectifyMap(KR,DR, R2, P2, DIM, cv2.CV_32FC1)

                # Read images
                # left = cv2.imread("balls/L_0.png")
                # right = cv2.imread("balls/R_0.png")

                # Setup SimpleBlobDetector parameters.
                params = cv2.SimpleBlobDetector_Params()

                # Change thresholds
                params.minThreshold = cv_params["minThreshold"]
                params.maxThreshold = cv_params["maxThreshold"]


                # Filter by Area.
                params.filterByArea = True
                params.minArea = cv_params["minArea"]
                params.maxArea = cv_params["maxArea"]

                # Filter by Circularity
                params.filterByCircularity = True
                params.minCircularity = cv_params["minCircularity"]

                # Filter by Convexity
                params.filterByConvexity = False
                params.minConvexity = cv_params["minConvexity"]
                    
                # Filter by Inertia
                params.filterByInertia = True
                params.minInertiaRatio = cv_params["minInertiaRatio"]

                # Create a detector with the parameters
                ver = (cv2.__version__).split('.')
                if int(ver[0]) < 3 :
                    detector = cv2.SimpleBlobDetector(params)
                else : 
                    detector = cv2.SimpleBlobDetector_create(params)

                lower_yellow = np.array(cv_params["lower_yellow"])
                upper_yellow = np.array(cv_params["upper_yellow"])

                while(True):
                    rclpy.spin_once(turtle_node)
                    if turtle_node.voltage < threshold:
                        print("voltage too low--powering off...")
                        turtle_node.Joints.disable_torque()
                        break
                        # print("traj 1...")
                    if turtle_node.mode == 'rest' or turtle_node.mode == 'stop':
                        turtle_node.Joints.send_torque_cmd(turtle_node.nq * [0])
                        turtle_node.Joints.disable_torque()
                        first_time = True
                        break

                    cap0 = cv2.VideoCapture(1)
                    cap1 = cv2.VideoCapture(2)

                    ret0, left = cap1.read()
                    ret1, right = cap0.read()
                    
                    fixedLeft = cv2.remap(left, left1, left2, cv2.INTER_LINEAR)
                    fixedRight = cv2.remap(right, right1, right2, cv2.INTER_LINEAR)
                    
                    # Converting from BGR to HSV color space
                    hsv = cv2.cvtColor(fixedRight, cv2.COLOR_BGR2HSV)
                    
                    # Compute mask
                    premask = cv2.inRange(hsv, lower_yellow, upper_yellow)
                    mask = cv2.bitwise_not(premask)
                    
                    # Bitwise AND 
                    # result = cv2.bitwise_and(im,im, mask= mask)

                    # cv2.imshow('Mask',mask)
                    # cv2.waitKey(0)
                    # cv2.imshow('Masked Image',result)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()

                    # Detect blobs.

                    keypoints = detector.detect(mask)
                    if not (len(keypoints) == 0):
                        centroid = (keypoints[0].pt[0], keypoints[0].pt[1])
                        print(centroid)
                    # Determine largest blob (target) and centroid 
                    #target_blob = keypoints
                    # Draw detected blobs as red circles.
                    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
                    # the size of the circle corresponds to the size of blob

                    im_with_keypoints = cv2.drawKeypoints(mask, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                    # Show blobs
                    cv2.imshow("Keypoints", im_with_keypoints)
                    if cv2.waitKey(1) == 27:
                        break
                    turn_thresh = cv_params["turn_thresh"]
                    dive_thresh = cv_params["dive_thresh"]
                    if abs(centroid[0] - 1920/2) < turn_thresh and abs(centroid[1] - 1080/2) < dive_thresh:
                        # output straight primitive
                        primitive = 'straight'
                    elif centroid[0] > 1920 - (1920/2 - turn_thresh):
                        # turn right
                        primitive = 'right'
                    elif centroid[0] < (1920/2 - turn_thresh):
                        # turn left
                        primitive = 'left'
                    elif centroid[1] > 1080 - (1080/2 - dive_thresh):
                        # dive
                        primitive = 'dive'
                    elif centroid[1] < (1080/2 - dive_thresh): 
                        # surface
                        primitive = 'surface'
                    else:
                        print("dwell")
                        # dwell
                        primitive = 'none'
                    prev_primitive = 'none'
                    if primitive == 'none':
                        turtle_node.Joints.disable_torque()
                    else:
                        num_cycles = cv_params["num_cycles"]
                        turtle_node.Joints.enable_torque()
                        # run the given primitive
                        qd_mat = mat2np(f'/home/crush/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/turtle_trajectory/{primitive}/qd.mat', 'qd')
                        dqd_mat = mat2np(f'/home/crush/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/turtle_trajectory/{primitive}/dqd.mat', 'dqd')
                        ddqd_mat = mat2np(f'/home/crush/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/turtle_trajectory/{primitive}/ddqd.mat', 'ddqd')
                        tvec = mat2np(f'/home/crush/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/turtle_trajectory/{primitive}/tvec.mat', 'tvec')
                        # if prev_primitive != primitive
                        first_time = True
                        first_loop = True
                        input_history = np.zeros((turtle_node.nq,10))
                        q_data = np.zeros((turtle_node.nq,1))
                        tau_data = np.zeros((turtle_node.nq,1))
                        timestamps = np.zeros((1,1))
                        dq_data = np.zeros((turtle_node.nq,1))
                        tau_data = np.zeros((turtle_node.nq,1))
                        timestamps = np.zeros((1,1))
                        dt_loop = np.zeros((1,1))       # hold dt data 
                        cycle = 0

                        # zero =  np.zeros((self.nq,1))
                        t_old = time.time()
                        # our loop's "starting" time
                        t_0 = time.time()
                        while cycle < num_cycles:
                            rclpy.spin_once(turtle_node)
                            if turtle_node.voltage < threshold:
                                print("voltage too low--powering off...")
                                turtle_node.Joints.disable_torque()
                                break
                            if turtle_node.mode == 'rest' or turtle_node.mode == 'stop':
                                turtle_node.Joints.send_torque_cmd(turtle_node.nq * [0])
                                turtle_node.Joints.disable_torque()
                                first_time = True
                                break
                            
                            q = np.array(turtle_node.Joints.get_position()).reshape(-1,1)
                            if first_loop:
                                n = get_qindex((time.time() - t_0), tvec)
                            else:
                                offset = t_0 - 2
                                n = get_qindex((time.time() - offset), tvec)

                            if n == len(tvec[0]) - 1:
                                first_loop = False
                                t_0 = time.time()
                                cycle += 1
                                print(f"-----------------cycle: {cycle}\n\n\n")
                            
                            qd = np.array(qd_mat[:, n]).reshape(-1,1)
                            dqd = np.array(dqd_mat[:, n]).reshape(-1,1)
                            ddqd = np.array(ddqd_mat[:, n]).reshape(-1,1)
                            q_data=np.append(q_data, q, axis = 1) 
                            
                            if first_time:
                                dq = np.array(turtle_node.Joints.get_velocity()).reshape(-1,1)
                                dq_data=np.append(dq_data, dq, axis = 1) 
                                first_time = False
                            else:
                                t = time.time()
                                dt = t - t_old
                                t_old = t
                                dq = np.array(turtle_node.Joints.get_velocity()).reshape(-1,1)
                                dq_data=np.append(dq_data, dq, axis = 1) 
                                # # calculate errors
                            tau = turtle_controller(q,dq,qd,dqd,ddqd,turtle_node.Kp,turtle_node.KD)
                            
                            input_history = np.append(input_history[:,1:], tau,axis=1)
                            input_mean = np.mean(input_history, axis = 1)
                            curr = grab_arm_current(input_mean, min_torque, max_torque)
                            turtle_node.Joints.send_torque_cmd(curr)
                            prev_primitive = primitive 
                    
                cv2.destroyAllWindows()

            elif turtle_node.mode == 'planner':
                """
                Randomly pick a motion primitive and run it 4-5 times
                """
                primitives = ['surface', 'turnrf', 'turnrr', 'straight', 'turnlr']
                # primitives = ['turnrr']
                num_cycles = 4
                turtle_node.Joints.disable_torque()
                turtle_node.Joints.set_current_cntrl_mode()
                turtle_node.Joints.enable_torque()

                while True:
                    rclpy.spin_once(turtle_node)
                    if turtle_node.mode == 'rest' or turtle_node.mode == 'stop' or turtle_node.voltage < threshold:
                        turtle_node.Joints.send_torque_cmd([0] *len(turtle_node.IDs))
                        turtle_node.Joints.disable_torque()
                        break
                    primitive = random.choice(primitives)
                    print(f"---------------------------------------PRIMITIVE: {primitive}\n\n")
                    qd_mat = mat2np(f'/home/crush/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/turtle_trajectory/{primitive}/qd.mat', 'qd')
                    dqd_mat = mat2np(f'/home/crush/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/turtle_trajectory/{primitive}/dqd.mat', 'dqd')
                    ddqd_mat = mat2np(f'/home/crush/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/turtle_trajectory/{primitive}/ddqd.mat', 'ddqd')
                    tvec = mat2np(f'/home/crush/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/turtle_trajectory/{primitive}/tvec.mat', 'tvec')
                    first_time = True
                    first_loop = True
                    input_history = np.zeros((turtle_node.nq,10))
                    q_data = np.zeros((turtle_node.nq,1))
                    tau_data = np.zeros((turtle_node.nq,1))
                    timestamps = np.zeros((1,1))
                    dt_loop = np.zeros((1,1))       # hold dt data 
                    dq_data = np.zeros((turtle_node.nq,1))
                    tau_data = np.zeros((turtle_node.nq,1))
                    timestamps = np.zeros((1,1))
                    dt_loop = np.zeros((1,1))       # hold dt data 
                    cycle = 0

                    # zero =  np.zeros((self.nq,1))
                    t_old = time.time()
                    # our loop's "starting" time
                    t_0 = time.time()
                    while cycle < num_cycles:
                        if turtle_node.voltage < threshold:
                            print("voltage too low--powering off...")
                            turtle_node.Joints.disable_torque()
                            break
                        rclpy.spin_once(turtle_node)
                        # print("traj 1...")
                        if turtle_node.mode == 'rest' or turtle_node.mode == 'stop':
                            turtle_node.Joints.send_torque_cmd(turtle_node.nq * [0])
                            turtle_node.Joints.disable_torque()
                            first_time = True
                            break
                        
                        q = np.array(turtle_node.Joints.get_position()).reshape(-1,1)
                        if first_loop:
                            n = get_qindex((time.time() - t_0), tvec)
                        else:
                            # print("done with first loop")
                            offset = t_0 - 2
                            n = get_qindex((time.time() - offset), tvec)

                        # print(f"n: {n}\n")
                        if n == len(tvec[0]) - 1:
                            first_loop = False
                            t_0 = time.time()
                            cycle += 1
                            print(f"-----------------cycle: {cycle}\n\n\n")
                        
                        qd = np.array(qd_mat[:, n]).reshape(-1,1)
                        dqd = np.array(dqd_mat[:, n]).reshape(-1,1)
                        ddqd = np.array(ddqd_mat[:, n]).reshape(-1,1)
                        # # print(f"[DEBUG] qdata: {q_data}\n")
                        # print(f"[DEBUG] qd: {qd}\n")
                        q_data=np.append(q_data, q, axis = 1) 
                        # # At the first iteration velocity is 0  
                        
                        if first_time:
                            # dq = np.zeros((nq,1))
                            dq = np.array(turtle_node.Joints.get_velocity()).reshape(-1,1)
                            dq_data=np.append(dq_data, dq, axis = 1) 
                            q_old = q
                            first_time = False
                        else:
                            t = time.time()
                            dt = t - t_old
                        #     # print(f"[DEBUG] dt: {dt}\n")  
                            t_old = t
                            # dq = diff(q, q_old, dt)
                            dq = np.array(turtle_node.Joints.get_velocity()).reshape(-1,1)
                            dq_data=np.append(dq_data, dq, axis = 1) 
                            q_old = q
                            # # calculate errors
                        err = q - qd
                        # # print(f"[DEBUG] e: {err}\n")
                        # # print(f"[DEBUG] q: {q * 180/3.14}\n")
                        # # print(f"[DEBUG] qd: {qd * 180/3.14}\n")
                        err_dot = dq

                        tau = turtle_controller(q,dq,qd,dqd,ddqd,turtle_node.Kp,turtle_node.KD)

                        # publish motor data, tau data, 
                        
                        input_history = np.append(input_history[:,1:], tau,axis=1)
                        input_mean = np.mean(input_history, axis = 1)
                        curr = grab_arm_current(input_mean, min_torque, max_torque)
                        turtle_node.Joints.send_torque_cmd(curr)

                turtle_node.Joints.disable_torque()

            else:
                print("wrong command received....")

    except Exception as e:
        print("some error occurred")
        turtle_node.Joints.send_torque_cmd([0] * len(turtle_node.IDs))
        turtle_node.Joints.disable_torque()
        exec_type, obj, tb = sys.exc_info()
        fname = os.path.split(tb.tb_frame.f_code.co_filename)[1]
        print(exec_type, fname, tb.tb_lineno)
        print(e)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

