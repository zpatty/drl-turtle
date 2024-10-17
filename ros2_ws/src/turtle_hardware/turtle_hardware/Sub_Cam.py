import rclpy
from rclpy.node import Node 
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import String
from sensor_msgs.msg import Image, CompressedImage
from turtle_interfaces.msg import TurtleCam, TurtleMode
import cv2
from cv_bridge import CvBridge
import cv2 
from matplotlib import pyplot as plt
import time
import os
from datetime import datetime
import sys
import numpy as np

class CamSubscriber(Node):

    def __init__(self):
        super().__init__('cam_sub_node')
        self.flag = ''
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=2
        )
        self.frames_sub = self.create_subscription(
            TurtleCam,
            'frames',
            self.img_callback,
            qos_profile
            )


        self.cam_depth = self.create_subscription(
            CompressedImage,
            'video_frames_depth',
            self.img_callback_depth,
            qos_profile
            )
    

        self.cam_detect = self.create_subscription(
            CompressedImage,
            'video_detect',
            self.img_callback_detect,
            qos_profile
            )
        
        self.mode_sub = self.create_subscription(
            TurtleMode,
            'turtle_mode',
            self.turtle_mode_callback,
            qos_profile)

        self.create_rate(1000)
        self.br = CvBridge()
        self.frames = []
        self.count = 0
        self.detect_count = 0
        self.depth_count = 0
        self.first = 1
        self.start_time = time.time()
        t = datetime.today().strftime("%m_%d_%Y_%H_%M_%S")
        folder_name =  "video/" + t
        os.makedirs(folder_name)
        self.output_folder = folder_name

    def img_callback(self, msg):
        self.get_logger().info('Receiving video frame')
        left = self.br.compressed_imgmsg_to_cv2(msg.data[0])
        right = self.br.compressed_imgmsg_to_cv2(msg.data[1])
        fused = np.concatenate((left[:,0:230], right), axis=1)
        cv2.imwrite(self.output_folder + "/left/frame%d.jpg" % self.count, left)
        cv2.imwrite(self.output_folder + "/right/frame%d.jpg" % self.count, right)
        self.count += 1
        end_time = time.time()
        seconds = end_time - self.start_time
        fps = 1.0 / seconds
        print("Estimated frames per second : {0}".format(fps))
        self.start_time = end_time
        cv2.imshow("fused", fused)
        cv2.waitKey(1)


    def img_callback_detect(self, data):
        self.destroy_subscription(self.frames_sub)
        cv2.destroyWindow("fused")
        self.get_logger().info('Receiving video frame')
        # print(data.shape)
        current_frame = self.br.compressed_imgmsg_to_cv2(data)
        # shesh = '/home/ranger/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/'
        # cv2.imwrite(shesh + "images/frame%d.jpg" % self.count, current_frame)
        cv2.imwrite(self.output_folder + "/detection/frame%d.jpg" % self.count, current_frame)
        self.detect_count += 1
        # print(current_frame.shape)
        end_time = time.time()
        seconds = end_time - self.start_time
        fps = 1.0 / seconds
        print("Estimated frames per second : {0}".format(fps))
        self.start_time = end_time
        cv2.imshow("detection", current_frame)   
        cv2.waitKey(1)

    def img_callback_depth(self, data):
        self.get_logger().info('Receiving other video frame')
        current_frame = self.br.compressed_imgmsg_to_cv2(data)
        # shesh = '/home/ranger/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/'
        # cv2.imwrite(shesh + "images/frame%d.jpg" % self.count, current_frame)
        # self.count += 1
        cv2.imwrite(self.output_folder + "/depth/frame%d.jpg" % self.count, current_frame)
        self.depth_count += 1
        if self.first:
            plt.ion()
            self.fig, ax = plt.subplots()
            self.im = ax.imshow(current_frame)   
            plt.show()
            self.first = 0
        else:
            self.im.set_data(current_frame)
            self.fig.canvas.flush_events()
    
    def turtle_mode_callback(self, msg):
        if msg.mode == "kill":
            raise KeyboardInterrupt


def main(args=None):
    rclpy.init(args=args)
    print("Cowabunga")
    cam_sub = CamSubscriber()
    try:
        rclpy.spin(cam_sub)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        # rclpy.shutdown()
        print("some error occurred")
        # turtle_node.shutdown_motors()
        exec_type, obj, tb = sys.exc_info()
        fname = os.path.split(tb.tb_frame.f_code.co_filename)[1]
        print(exec_type, fname, tb.tb_lineno)
        print(e)
    cv2.destroyAllWindows() 
if __name__ == '__main__':
  main()