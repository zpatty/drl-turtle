# import rclpy
# from rclpy.node import Node 
# from sensor_msgs.msg import Image
# from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
# from threading import Thread
# from cv_bridge import CvBridge
# import cv2 
    
################################################## on the raspi4
# class CamStream():
#     """
#     Class that just reads from the camera
#     """
#     def __init__(self, idx=0):
#         self.cap = cv2.VideoCapture(0)
#         self.ret, self.frame = self.cap.read()
#         self.stopped = False
    
#     def start(self):
#         Thread(target=self.get, args=()).start()
#         return self
#     def get(self):
#         while not self.stopped:
#             self.ret, self.frame = self.cap.read()
#     def stop_process(self):
#         self.stopped = True

# class CamProcessor():
#     def __init__(self):
#         self.stream = None
    
#     def start(self):
#         Thread(target=self.proc, args=()).start()
#         return self

#     def set_stream(self, cam_stream):
#         self.stream = cam_stream
    
#     def proc(self):
#         while not self.stopped:
#             if self.stream is not None:
#                 frame = self.stream.frame

# class CamPublisher(Node):
#     def __init__(self):
#         super().__init__('cam_pub_node')

#         topic_name= 'video_frames'
#         self.br = CvBridge()
#         qos_profile = QoSProfile(
#             reliability=ReliabilityPolicy.BEST_EFFORT,
#             durability=DurabilityPolicy.TRANSIENT_LOCAL,
#             history=HistoryPolicy.KEEP_LAST,
#             depth=50
#         )
#         self.publisher_ = self.create_publisher(Image, topic_name , qos_profile)
#         self.create_rate(100)
# #         self.timer = self.create_timer(0.02, self.timer_callback)

# #     def timer_callback(self):
# #         ret, frame = self.cap.read()     
# #         if ret == True:
# #             self.publisher_.publish(self.br.cv2_to_imgmsg(frame, encoding="passthrough"))
# #         self.get_logger().info('Publishing video frame')

# def main(args=None):
#     rclpy.init(args=args)
#     cam_pub = CamPublisher()
#     # try:
#     #     rclpy.spin(cam_pub)
#     # except:
#     #     cap0.release()
#     #     cam_pub.destroy_node()
#     #     rclpy.shutdown()
#     stream = CamStream().start()
#     # proc = CamProcessor().start()
#     # proc.set_stream(stream)

#     try:
#         while True:
#             rclpy.spin_once(cam_pub)
#             cam_pub.publisher_.publish(cam_pub.br.cv2_to_imgmsg(stream.frame, encoding="passthrough"))
#             cam_pub.get_logger().info('Publishing video frame')
#     except KeyboardInterrupt:
#         stream.stop_process()
#         cam_pub.destroy_node()
#         rclpy.shutdown()

  
# if __name__ == '__main__':
#   main()




import rclpy
from rclpy.node import Node 
from sensor_msgs.msg import Image
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from threading import Thread
from std_msgs.msg import String
from rclpy.executors import MultiThreadedExecutor
from cv_bridge import CvBridge
import cv2 
    
class CamStream():
    """
    Class that just reads from the camera
    """
    def __init__(self, idx=0):
        self.cap = cv2.VideoCapture(0)
        self.ret, self.frame = self.cap.read()
        self.stopped = False
    
    def start(self):
        Thread(target=self.get, args=()).start()
        return self
    def get(self):
        while not self.stopped:
            self.ret, self.frame = self.cap.read()
    def stop_process(self):
        self.stopped = True
        self.cap.release()

class CamProcessor():
    def __init__(self):
        self.stream = None
    
    def start(self):
        Thread(target=self.proc, args=()).start()
        return self

    def set_stream(self, cam_stream):
        self.stream = cam_stream
    
    def proc(self):
        while not self.stopped:
            if self.stream is not None:
                frame = self.stream.frame

class CamPublisher(Node):
    def __init__(self):
        super().__init__('cam_pub_node')

        topic_name= 'video_frames'
        self.br = CvBridge()
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.publisher_ = self.create_publisher(Image, topic_name , qos_profile)
        self.create_rate(50)
        # self.cap = cv2.VideoCapture(0)
        # self.ret, self.frame = self.cap.read()

    #     self.timer = self.create_timer(0.03, self.timer_callback)
    # def timer_callback(self):
    #     ret, frame = self.cap.read()     
    #     if ret == True:
    #         self.publisher_.publish(self.br.cv2_to_imgmsg(frame, encoding="passthrough"))
    #     self.get_logger().info('Publishing video frame')


def main(args=None):
    rclpy.init(args=args)
    cam_pub = CamPublisher()
    stream = CamStream().start()
    denoise = 15

    try:
        while True:
            rclpy.spin_once(cam_pub)
            cam_pub.publisher_.publish(cam_pub.br.cv2_to_imgmsg(stream.frame, encoding="passthrough"))
            cam_pub.get_logger().info('Publishing video frame')
    except KeyboardInterrupt:
        print("keyboard interrupt")
        stream.stop_process()
        cam_pub.destroy_node()
        rclpy.shutdown()


  
if __name__ == '__main__':
    main()
