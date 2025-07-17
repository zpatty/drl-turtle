#!/bin/bash
sleep 1
cd drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/ ;
source ~/colcon_venv/venv/bin/activate ;
python3 turtle_tracker.py ;