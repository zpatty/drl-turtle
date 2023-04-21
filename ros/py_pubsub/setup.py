from setuptools import setup

package_name = 'py_pubsub'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='gracexu@mit.edu',
    description='publish subscribe',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': ['motors=py_pubsub.motors:main', 'camera=py_pubsub.camera:main',
'control=py_pubsub.control:main', 'sensors=py_pubsub.sensors:main', 'comms=py_pubsub.comms:main', 'planner=py_pubsub.planner:main',
        ],
    },
)
