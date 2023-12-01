from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='py_pubsub',
            executable='master',
            name='master'
        ),
        Node(
            package='py_pubsub',
            executable='motors',
            name='motors'
        ),
        Node(
            package='py_pubsub',
            executable='cv',
            name='cv'
        ),
        Node(
            package='py_pubsub',
            executable='sonar',
            name='sonar'
        )
    ])