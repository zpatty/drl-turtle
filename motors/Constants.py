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
TKEY_ASCII_VALUE            = 0x74     
MOD1_VALUE                  = 0x31      # pressing 1 on keyboard
MOD2_VALUE                  = 0x32
MOD3_VALUE                  = 0x33

# ID that commands all dynamixels
DXL_IDALL                   = 254
# Protocol version
PROTOCOL_VERSION            = 2.0            # See which protocol version is used in the Dynamixel
MOD_DEVICE                  = '/dev/ttyUSB0'
JOINTS                      = '/dev/ttyUSB1'
BAUDRATE                    = 2000000
portHandlerMod              = PortHandler(MOD_DEVICE)
packetHandlerMod            = PacketHandler(PROTOCOL_VERSION)
portHandlerJoint            = PortHandler(JOINTS)
packetHandlerJoint          = PacketHandler(PROTOCOL_VERSION)
MAX_VELOCITY = 20

# some constants 
d = 0.04337                    # circumradius of hexagon plate (m)
r = 0.004                      # spool radius (m)
err = math.inf                 # start with large error
mass_module = .15              # in grams
mj = mass_module/4
m = mj
mplate = 0.08          # in kilograms
hp = 0.007
Lm = 0.013             # distance between center of spool and centroid of bottom plate
# lengths when modules are not stressed (in m)
l1_0 = [0.065, 0.066, 0.0645]
l2_0 = [0.0785, 0.0775, 0.077]
l3_0 = [0.082, 0.0815, 0.0815]
l4_0 = [0.082, 0.0815, 0.0815]


limits = [l1_0, l2_0, l3_0, l4_0]
# our reference motor angles (when arm is neutral, a.k.a standing)
th1_0 = [-3.179942173286934, -6.13745713233045, -4.693981210930062]
th2_0 = [5.333651199478374, 1.363708920430335, 1.1642914180052018]
th3_0 = [0.7086991240031663, -14.473108733701025, 4.451612246444131]
th4_0 = [0.7086991240031663, -14.473108733701025, 4.451612246444131]

# our joint limits (this is without subtracting the offset)
min_joint_theta = 1.99
max_joint_theta = 4.27

upper_limit = l1_0
lower_limit  = l1_min
# min and max are in mA because Dynamixel takes in mA inputs for current control
max_torque = 75
xm_max_torque = 4500
min_torque = 5
xm_min_torque = 2