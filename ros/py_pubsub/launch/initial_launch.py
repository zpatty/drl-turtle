from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='py_pubsub',
            executable='control',
            name='control'
        ),
        Node(
            package='py_pubsub',
            executable='motors',
            name='motors'
        )
    ])