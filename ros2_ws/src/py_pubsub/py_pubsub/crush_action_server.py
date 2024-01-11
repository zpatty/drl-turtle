import time

import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node

from my_package.action import Crush


class CrushActionServer(Node):

    def __init__(self):
        super().__init__('crush_action_server')
        self._action_server = ActionServer(
            self,
            Crush,
            'crush',
            self.execute_callback)
        print("initializing...")

    def execute_callback(self, goal_handle):
        """
        Method that will be called to execute goal once it is accepted 
        """
        self.get_logger().info('Executing goal...')
        feedback_msg = Crush.Feedback()
        feedback_msg.partial_traj = []
        traj_type = goal_handle.request.mode
        traj = []
        if traj_type == 5:
            traj = [0, 1, 2, 3]
            for i in traj:
                feedback_msg.partial_traj.append(i)
                self.get_logger().info('Feedback: {0}'.format(feedback_msg.partial_traj))
                goal_handle.publish_feedback(feedback_msg)
                time.sleep(1)

        else:
            traj = [0]
        
        goal_handle.succeed()
        result = Crush.Result()
        result.traj = traj
        return result


def main(args=None):
    rclpy.init(args=args)

    crush_action_server = CrushActionServer()

    rclpy.spin(crush_action_server)


if __name__ == '__main__':
    main()