#!/bin/bash
sleep 3
cd drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/ ;
source ~/colcon_venv/venv/bin/activate ;
python3 TurtleController.py ;