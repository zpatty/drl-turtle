#!/usr/bin/env python
# -*- coding: utf-8 -*-
from dynamixel_sdk import *                    # Uses Dynamixel SDK library
from Dynamixel import *                        # Dynamixel motor class
import math
# KEYBOARD INPUTS
ESC_ASCII_VALUE             = 0x1b
SPACE_ASCII_VALUE           = 0x20
WKEY_ASCII_VALUE            = 0x77
SKEY_ASCII_VALUE            = 0x73
AKEY_ASCII_VALUE            = 0x61
DKEY_ASCII_VALUE            = 0x64
CKEY_ASCII_VALUE            = 0x63
BKEY_ASCII_VALUE            = 0x62      # key to bend the top module
UKEY_ASCII_VALUE            = 0x75      # key to unbend the modules
NKEY_ASCII_VALUE            = 0x6E
IKEY_ASCII_VALUE            = 0x69     
QKEY_ASCII_VALUE            = 0x71     

# ID that commands all dynamixels
DXL_IDALL                   = 254
# Protocol version
PROTOCOL_VERSION            = 2.0            # See which protocol version is used in the Dynamixel
MOD_DEVICE                  = 'COM4'
JOINTS                      = '/dev/ttyUSB0'
BAUDRATE                    = 57600
portHandlerMod              = PortHandler(MOD_DEVICE)
packetHandlerMod            = PacketHandler(PROTOCOL_VERSION)
MAX_VELOCITY = 20

# some constants 
d = 0.04337                    # circumradius of hexagon plate (m)
r = 0.003                      # spool radius (m)
err = math.inf                 # start with large error
mass_module = 105              # in grams
mj = mass_module/5
m = [mj, mj, mj, mj, mj, mj, mj, mj, mj]
mplate = 0.015          # in kilograms
hp = 0.007
# lengths when modules are not stressed (in m)
l1_0 = [0.078, 0.078, 0.0765]
# our reference motor angles (when arm is neutral, a.k.a standing)
th1_0 = [11.963516164720115, 6.557767868211116, 5.675728915176872]
# absolute minimum lengths motors can handle (must measure this out using grab_cable_lengths.py)
l1_min = [0.0287, 0.0302, 0.0291]
# our joint limits (this is without subtracting the offset)
min_joint_theta = 1.99
max_joint_theta = 4.27
# offset is the angle of the joint when the arm is standing (neutral pos)
offset = 3.14

upper_limit = l1_0
lower_limit  = l1_min
# min and max are in mA because Dynamixel takes in mA inputs for current control
max_torque = 75
min_torque = 5