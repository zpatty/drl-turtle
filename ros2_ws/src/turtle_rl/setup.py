from setuptools import find_packages, setup

package_name = 'turtle_rl'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Emily Sologuren',
    maintainer_email='ersologuren@gmail.com',
    description='Package for all RL turtle algorithms',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'CPG_node = turtle_rl.CPG_node:main'
        ],
    },
)
