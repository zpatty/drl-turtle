import rclpy
from rclpy.node import Node 
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2 
from matplotlib import pyplot as plt
class CamSubscriber(Node):

    def __init__(self):
        super().__init__('cam_sub_node')
        self.flag = ''
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=50
        )
        self.cam_subscription = self.create_subscription(
            Image,
            'video_frames',
            self.img_callback,
            qos_profile
            )
        # self.cam_subscription_1 = self.create_subscription(
        # Image,
        # 'video_frames_1',
        # self.img_callback_1,
        # qos_profile
        # )

        self.cam_color = self.create_subscription(
            Image,
            'video_frames_color',
            self.img_callback_color,
            qos_profile
            )

        # self.cam_color_1 = self.create_subscription(
        #     Image,
        #     'video_frames_color_1',
        #     self.img_callback_color_1,
        #     qos_profile
        #     )

        self.cam_depth = self.create_subscription(
            Image,
            'video_frames_depth',
            self.img_callback_depth,
            qos_profile
            )


        self.key_subscription = self.create_subscription(
            String,
            'turtle_mode_cmd',
            self.keyboard_callback,
            qos_profile,
            )
        
        self.create_rate(100)
        self.br = CvBridge()
        self.frames = []
        self.count = 0

    def img_callback(self, data):
        self.get_logger().info('Receiving video frame')
        current_frame = self.br.imgmsg_to_cv2(data)
        # shesh = '/home/ranger/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/'
        # cv2.imwrite(shesh + "images/frame%d.jpg" % self.count, current_frame)
        # self.count += 1
        cv2.imshow("mask", current_frame)   
        cv2.waitKey(1)

    # def img_callback_1(self, data):
    #     self.get_logger().info('Receiving other video frame')
    #     current_frame = self.br.imgmsg_to_cv2(data)
    #     # shesh = '/home/ranger/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/'
    #     # cv2.imwrite(shesh + "images/frame%d.jpg" % self.count, current_frame)
    #     # self.count += 1
    #     cv2.imshow("camera_1", current_frame)   
    #     cv2.waitKey(1)
    
    def img_callback_color(self, data):
        self.get_logger().info('Receiving other video frame')
        current_frame = self.br.imgmsg_to_cv2(data)
        # shesh = '/home/ranger/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/'
        # cv2.imwrite(shesh + "images/frame%d.jpg" % self.count, current_frame)
        # self.count += 1
        cv2.imshow("color", current_frame)   
        cv2.waitKey(1)

    # def img_callback_color_1(self, data):
    #     self.get_logger().info('Receiving other video frame')
    #     current_frame = self.br.imgmsg_to_cv2(data)
    #     # shesh = '/home/ranger/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/'
    #     # cv2.imwrite(shesh + "images/frame%d.jpg" % self.count, current_frame)
    #     # self.count += 1
    #     cv2.imshow("left camera", current_frame)   
    #     cv2.waitKey(1)

    def img_callback_depth(self, data):
        self.get_logger().info('Receiving other video frame')
        current_frame = self.br.imgmsg_to_cv2(data)
        # shesh = '/home/ranger/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/'
        # cv2.imwrite(shesh + "images/frame%d.jpg" % self.count, current_frame)
        # self.count += 1
        cv2.imshow("depth", current_frame)   
        cv2.waitKey(1)

    def keyboard_callback(self, msg):
        """
        Callback function that updates the mode the turtle should be in.
        This method is what enables us to set "emergency stops" mid-trajectory. 
        """
        # global mode
        if msg.data == 'stop':
            self.flag = 'stop'

def main(args=None):
    rclpy.init(args=args)
    print("yo")
    cam_sub = CamSubscriber()
    while(cam_sub.flag != 'stop'):
        rclpy.spin_once(cam_sub)
    cam_sub.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
  main()