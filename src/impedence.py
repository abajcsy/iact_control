#! /usr/bin/env python
"""
This node demonstrates simple torque-based impedance control by moving the Jaco
so that it maintains a fixed distance to a target. 

Author: Andrea Bajcsy (abajcsy@eecs.berkeley.edu)
Based on: https://w3.cs.jmu.edu/spragunr/CS354_S15/labs/pid_lab/pid_lab.shtml
"""
import roslib; roslib.load_manifest('kinova_demo')

import rospy
import math
import pid
import tf
import sys
import thread
import argparse
import actionlib
import time
import plot

import kinova_msgs.msg
import geometry_msgs.msg
import std_msgs.msg
from kinova_msgs.srv import *
from std_msgs.msg import Float32

import numpy as np
import matplotlib.pyplot as plt

prefix = 'j2s7s300_driver'

class Impedence(object):
	"""
	This class represents a node that moves the Jaco with impedence control.
	The joint torques are computed as:
		
		T = J^T(K(e) + B(edot))

	where:

		e = (target_joint configuration) - (current joint configuration)
		e_dot = derivative of error
		K = end-effector stiffness
		B = damping
	"""
	def __init__(self, k_gain, b_gain):
	
	def update_Impedence(self, p_error, dt=None):
		
	