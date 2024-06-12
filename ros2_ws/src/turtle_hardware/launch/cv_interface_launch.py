import launch
import launch_ros.actions

def generate_launch_description():
    return launch.LaunchDescription([
        launch_ros.actions.Node(
            package='turtle_hardware',
            executable='keyboard_node',
            name='keyboard_node'),
  ])
