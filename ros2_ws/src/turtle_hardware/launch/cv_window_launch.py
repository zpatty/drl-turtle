import launch
import launch_ros.actions

def generate_launch_description():
    return launch.LaunchDescription([
        launch_ros.actions.Node(
            package='turtle_hardware',
            executable='cam_subscriber_node',
            name='cam_subscriber_node'),
  ])

# from launch import LaunchDescription
# from launch_ros.actions import Node

# def generate_launch_description():
#     return LaunchDescription([
#         Node(
#             package='turtle_hardware',
#             namespace='turtle_interface',
#             executable='keyboard_node',
#             name='keyboard_node'
#         )

#     ])
