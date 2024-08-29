from setuptools import find_packages, setup

package_name = 'turtle_hardware'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/turtle_hardware/launch', ['launch/cv_launch.py', 'launch/cv_window_launch.py', 'launch/cv_interface_launch.py', 'launch/turtle_launch.py'])

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Emily Sologuren',
    maintainer_email='ersologuren@gmail.com',
    description='Turtle package for motors and sensors',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'turtle_ctrl_node = turtle_hardware.TurtleController:main',
            'log_node = turtle_hardware.Logger:main',
            'turtle_hardware_node = turtle_hardware.TurtleRobot:main',
            'turtle_sensors_node = turtle_hardware.TurtleSensorsNode:main',
            'turtle_tracker = turtle_hardware.turtle_tracker:main',
            'turtle_motor = turtle_hardware.main:main',
            'dual_cv_fused = turtle_hardware.dual_cv_fused:main',
        ],
    },
)
