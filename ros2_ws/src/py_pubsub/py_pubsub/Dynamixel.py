#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Below is a class that defines a Dynamixel and holds several useful methods

"""
ADDR_POS_D_GAIN             = 80
ADDR_POS_I_GAIN             = 82
ADDR_POS_P_GAIN             = 84
ADDR_TORQUE_ENABLE          = 64
ADDR_GOAL_CURRENT           = 102
ADDR_GOAL_POSITION          = 116
ADDR_GOAL_VELOCITY          = 104
ADDR_PRESENT_POSITION       = 132
ADDR_PRESENT_VELOCITY       = 128
ADDR_PROFILE_VELOCITY       = 112
ADDR_OPERATING_MODE         = 11
EXT_POSITION_CONTROL_MODE   = 4                 # The value of Extended Position Control Mode that can be set through the Operating Mode (11)
CURRENT_CONTROL_MODE        = 0                 # The value of Current Control Mode -- controls current(torque) regardless of speed and position 
VELOCITY_CONTROL_MODE       = 1                 # The value of Velocity Control Mode 
TORQUE_ENABLE               = 1                 # Value for enabling the torque
TORQUE_DISABLE              = 0                 # Value for disabling the torque
MAX_POSITION_VALUE          = 4095           # Of MX with firmware 2.0 and X-Series the revolution on Extended Position Control Mode is 256 rev
MIN_POSITION_VALUE          = 0          # important to note that this is for EXTENDED position mode
DXL_MOVING_STATUS_THRESHOLD = 20                # Dynamixel will rotate between this value
MAX_PROFILE_VELOCITY        = 50                # ranges from 0-32,767 [0.229 rev/min]

from dynamixel_sdk import *                    # Uses Dynamixel SDK library
# from packet_handling import *
import math
"""
can keep track of current length and the initial length it started with
"""

def twos_comp(bit_s):
    
    # print(f"our first bit: {bit_s[0]}")
    if bit_s[0] == '1':
        inverse_s = ''.join(['1' if i == '0' else '0'
                     for i in bit_s])
        new = ('1' + inverse_s)[2:]
        dec = -int(new,2)
        return dec
    else:
        return int(bit_s,2)

# print(twos_comp('11111111111111111111110100111000'))

def to_radians(steps):
    """
    Input: takes in dynamixel motor steps pos
    Output: radian value of dynamixel
    """
    rad_val = (steps * ((2 * math.pi)/4096))
    return rad_val

class Dynamixel:
    def __init__(self, packetHandler, portHandler, ID):
        self.packetHandler = packetHandler
        self.portHandler = portHandler
        self.ID            = ID
        print("We have initialized our dynamixel")

    def __str__(self):
        return f"This is motor: {self.ID}"

    def extended_pos_mode(self):
        # Set operating mode to extended position mode
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, self.ID, ADDR_OPERATING_MODE, EXT_POSITION_CONTROL_MODE)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else:
            print("Operating mode changed to extended position control mode.")
    def current_control_mode(self):
        # Set operating mode to current control mode (used for torque control)
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, self.ID, ADDR_OPERATING_MODE, CURRENT_CONTROL_MODE)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else:
            print("Operating mode changed to current control mode.")
    def set_max_velocity(self, velocity):
        # Set max velocity of dynamixel
        dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(self.portHandler, self.ID, ADDR_PROFILE_VELOCITY, velocity)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else:
            print("Max velocity has been set")
    def enable_torque(self):
        #Enable Dynamixel Torque
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, self.ID, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else:
            print("Dynamixel torque has been set")
    def disable_torque(self):
        # Disable Dynamixel Torque
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, self.ID, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else:
            print("Dynamixel torque has been set")
    def set_goal_position(self, goal_pos):
        # Write goal position
        dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(self.portHandler, self.ID, ADDR_GOAL_POSITION, goal_pos)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else:
            pass
            print("goal position has been set")
    def send_torque_cmd(self, torque):
        # Write current input --> in mA
        dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(self.portHandler, self.ID, ADDR_GOAL_CURRENT, torque)
        # if dxl_comm_result != COMM_SUCCESS:
        #     print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        # elif dxl_error != 0:
        #     print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        # else:
        #     pass
    
    def get_present_pos(self):
        # Read present position --> gives you steps/rev (360 deg = 4095)
        # dxl_present_position, dxl_comm_result, dxl_error = self.packetHandler.read4ByteTxRx(self.portHandler, self.ID, ADDR_PRESENT_POSITION)
        data, result, error = self.packetHandler.readTxRx(self.portHandler, self.ID, ADDR_PRESENT_POSITION, 4)
        data_read = DXL_MAKEDWORD(DXL_MAKEWORD(data[0], data[1]),
                                  DXL_MAKEWORD(data[2], data[3])) if (result == COMM_SUCCESS) else 0
        # data read is decimal value
        print(f"data: {data}\n")
        # print(f"data 1: {bin(data[0])}\n")
        # print(f"data 2: {bin(data[1])}\n")
        # print(f"data 3: {bin(data[2])}\n")
        # print(f"data 4: {bin(data[3])}\n")
        
        d3 = bin(data[3])[2:]
        # print(f"combine the bits together: {comb}\n")
        
        # print(f"we should get 4337: {check}")
        # check_d = twos_comp(data_read)
        # if your binary number starts with a 1, you know it's negative, therefore you need to convert
        # if result != COMM_SUCCESS:
        #     print("%s" % self.packetHandler.getTxRxResult(result))
        # elif error != 0:
        #     print("%s" % self.packetHandler.getRxPacketError(error))
        # else:
        #     pass
            # print(f"Present Position for motor {self.ID} is {dxl_present_position}")
        # print(f"What sdk wants to output: {data_read}\n")
        # print(f"what we will output: {check_d}\n")
        if d3[0] == '0':
            return data_read
        else:
            d0 = bin(data[0])[2:]
            d1 = bin(data[1])[2:]
            d2 = bin(data[2])[2:]
            comb = d3 + d2 + d1 + d0
            print(f"comb: {comb}\n")
            check = twos_comp(comb)
            print(f"pos: {check}\n")
            return check

    def get_present_velocity(self):
        # Read present velocity --> gives you rev/min therefore convert
        dxl_present_velocity, dxl_comm_result, dxl_error = self.packetHandler.read4ByteTxRx(self.portHandler, self.ID, ADDR_PRESENT_VELOCITY)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else:
            pass
            # print(f"Present Position for motor {self.ID} is {dxl_present_position}")
        return dxl_present_velocity
    def setPID(self, Kp, Ki, Kd):
        dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(self.portHandler, self.ID, ADDR_POS_P_GAIN, Kp)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else:
            print("The Kp has been set.")

        
        dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(self.portHandler, self.ID, ADDR_POS_I_GAIN, Ki)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else:
            print("The Ki has been set.")

        
        dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(self.portHandler, self.ID, ADDR_POS_D_GAIN, Kd)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else:
            print("The Kd has been set.")
