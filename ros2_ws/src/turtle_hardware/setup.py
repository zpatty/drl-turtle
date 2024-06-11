from setuptools import find_packages, setup
from glob import glob

package_name = 'turtle_hardware'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/turtle_hardware/launch', ['launch/cv_launch.py', 'launch/cv_window_launch.py', 'launch/cv_interface_launch.py'])
        # (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*')))
        # ('share/' + package_name + '/launch/', ['cv_launch.py'])
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
            'keyboard_node = turtle_hardware.new_keyboard:main',
            'cam_subscriber_node = turtle_hardware.Cam_Sub:main',
            'cam_cv_node = turtle_hardware.test_cv:main',
            'turtle_cv_node = turtle_hardware.cv_node:main'
        ],
    },
)
