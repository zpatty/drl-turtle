import launch
import launch_ros.actions

def generate_launch_description():
    return launch.LaunchDescription([
        launch_ros.actions.Node(
            package='turtle_hardware',
            executable='turtle_hardware_node',
            name='turtle_hardware_node'),
        launch_ros.actions.Node(
            package='turtle_hardware',
            executable='turtle_ctrl_node',
            name='turtle_ctrl_node'),
        launch_ros.actions.Node(
            package='turtle_hardware',
            executable='log_node',
            name='log_node'),
  ])

# from launch import LaunchDescription
# from launch_ros.actions import Node

# def generate_launch_description():
#     return LaunchDescription([
#         Node(
#             package='turtle_hardware',
#             namespace='turtle_cam_window',
#             executable='cam_subscriber_node',
#             name='cam_subscriber_node'
#         ),
#         Node(
#             package='turtle_hardware',
#             namespace='turtle_cv_planner',
#             executable='cam_cv_node',
#             name='cam_cv_node'
#         )

#     ])


        # Node(
        #     package='turtle_hardware',
        #     namespace='turtle_interface',
        #     executable='keyboard_node',
        #     name='keyboard_node'
        # ),