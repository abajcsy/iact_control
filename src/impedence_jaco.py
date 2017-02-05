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

class ImpedenceJaco(object):
	"""
	This class represents a node that moves the Jaco with impedence control.
	The joint torques are computed as:
		
		T = J^T(K(e) + B(edot))

	where:

		e = (target_joint configuration) - (current joint configuration)
		e_dot = derivative of error
		K = end-effector stiffness
		B = damping
		
	Subscribes to: 
		/j2s7s300_driver/out/joint_angles	- Jaco joint angles
	
	Publishes to:
		/j2s7s300_driver/in/joint_torques	- Jaco joint torques 
	
	Required parameters:
		k_gain, b_gain    - gain terms for the PID controller
		j0, ... , j6			  - goal configuration for joints 1-7 (in degrees)
	"""
	def __init__(self,k_gain,b_gain,j0,j1,j2,j3,j4,j5,j6):
	
		rospy.init_node("impedence_jaco")

		# create joint-torque publisher
		self.torque_pub = rospy.Publisher(prefix + '/in/joint_torques', kinova_msgs.msg.JointTorque, queue_size=1)

		# create subscriber to joint_angles
		rospy.Subscriber(prefix + '/out/joint_angles', kinova_msgs.msg.JointAngles, self.joint_angles_callback, queue_size=1)

		# convert target position from degrees (default) to radians 		
		self.target_pos = np.array([j0,j1,j2,j3,j4,j5,j6]).reshape((7,1))* (math.pi/180.0)

		self.max_torque = 10*np.eye(7)
		self.torque = np.eye(7) 
		
		self.k_gain = k_gain
		self.b_gain = b_gain
		
		self.controller = pid.PID(P,I,D,0,0)

		# stuff for plotting
		self.plotter = plot.Plotter(self.p_gain,self.i_gain,self.d_gain)
		
		# publish to ROS at 100hz
		r = rospy.Rate(100) 
		while not rospy.is_shutdown(): #and nb != 'q':
			#nb = getch.getche() # need to import 
			self.torque_pub.publish(self.torque_to_JointTorqueMsg()) 
			r.sleep()
			
		# plot the error over time after finished
		#self.plotter.plot_Impedence()
		
		
		def velocity_to_JointTorqueMsg(self):
			"""
			Returns a JointTorque Kinova msg from an array of torques
			"""
			jointCmd = kinova_msgs.msg.JointTorque()
			jointCmd.joint1 = self.torque[0][0];
			jointCmd.joint2 = self.torque[1][1];
			jointCmd.joint3 = self.torque[2][2];
			jointCmd.joint4 = self.torque[3][3];
			jointCmd.joint5 = self.torque[4][4];
			jointCmd.joint6 = self.torque[5][5];
			jointCmd.joint7 = self.torque[6][6];

			return jointCmd
			
			
		def Impedence_control(self, pos):
			"""
			Return a control torque based on Impedence control
			"""
			# deal with angle wraparound when computing difference
			error = -((self.target_pos - pos + math.pi)%(2*math.pi) - math.pi)
			
			return -self.controller.update_Impedence(error)

			
		def joint_angles_callback(self, msg):
			"""
			Reads the latest position of the robot and publishes an
			appropriate torque command to move the robot to the target
			"""
			# read the current joint angles from the robot
			pos_curr = np.array([msg.joint1,msg.joint2,msg.joint3,msg.joint4,msg.joint5,msg.joint6,msg.joint7]).reshape((7,1))

			# convert to radians
			pos_curr = pos_curr*(math.pi/180.0)

			# update torque from PID based on current position
			self.torque = self.Impedence_control(pos_curr)
