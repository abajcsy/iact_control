#! /usr/bin/env python
"""
This node demonstrates torque-based PID control by moving the Jaco
so that it maintains a fixed distance to a target. 

Author: Andrea Bajcsy (abajcsy@eecs.berkeley.edu)
Based on: https://w3.cs.jmu.edu/spragunr/CS354_S15/labs/pid_lab/pid_lab.shtml
"""
import roslib; roslib.load_manifest('kinova_demo')

import rospy
import math
import pid
import tf
import sys, select, os
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
#from robot_control_modules import *

import numpy as np
import matplotlib.pyplot as plt

prefix = 'j2s7s300_driver'

class PIDTorqueJaco(object): 
	"""
	This class represents a node that moves the Jaco with PID control.
	The joint torques are computed as:
		
		T = -K_p(e) - K_d(e_dot) - K_i*Integral(e)

	where:

		e = (target_joint configuration) - (current joint configuration)
		e_dot = derivative of error
		K_p = accounts for present values of position error
		K_i = accounts for past values of error, accumulates error over time
		K_d = accounts for possible future trends of error, based on current rate of change
	
	Subscribes to: 
		/j2s7s300_driver/out/joint_angles	- Jaco joint angles
	
	Publishes to:
		/j2s7s300_driver/in/joint_torques	- Jaco joint torques
	
	Required parameters:
		p_gain, i_gain, d_gain    - gain terms for the PID controller
		j0, ... , j6				  - goal configuration for joints 1-7 (in degrees)
	"""

	def __init__(self, p_gain, i_gain, d_gain, j0,j1,j2,j3,j4,j5,j6):
		"""
		Setup of the ROS node. Publishing computed torques happens at 100Hz.
		"""

		rospy.init_node("pid_torque_jaco")

		# switch robot to torque-control mode
		self.init_torque_mode()

		# create joint-torque publisher
		self.torque_pub = rospy.Publisher(prefix + '/in/joint_torque', kinova_msgs.msg.JointTorque, queue_size=1)

		# create subscriber to joint_angles
		rospy.Subscriber(prefix + '/out/joint_angles', kinova_msgs.msg.JointAngles, self.joint_angles_callback, queue_size=1)
		# create subscriber to joint_torques
		rospy.Subscriber(prefix + '/out/joint_torques', kinova_msgs.msg.JointTorque, self.joint_torques_callback, queue_size=1)

		# convert target position from degrees (default) to radians 		
		self.target_pos = np.array([j0,j1,j2,j3,j4,j5,j6]).reshape((7,1))* (math.pi/180.0)

		self.max_torque = 20*np.eye(7)
		# stores current COMMANDED joint torques
		self.torque = np.eye(7) 
		# stores current joint MEASURED joint torques
		self.joint_torques = np.zeros((7,1))

		print "PID Gains: " + str(p_gain) + ", " + str(i_gain) + "," + str(d_gain)

		self.p_gain = p_gain
		self.i_gain = i_gain
		self.d_gain = d_gain

		# P, I, D gains 
		P = self.p_gain*np.eye(7)
		I = self.i_gain*np.eye(7)
		D = self.d_gain*np.eye(7)

		self.controller = pid.PID(P,I,D,0,0)

		# stuff for plotting
		self.plotter = plot.Plotter(self.p_gain,self.i_gain,self.d_gain)

		# publish to ROS at 100hz
		r = rospy.Rate(100) 

		print "----------------------------------"
		print "Moving robot, press ENTER to quit:"
		while not rospy.is_shutdown():

			if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
				line = raw_input()
				break

			self.torque_pub.publish(self.torque_to_JointTorqueMsg()) 
			r.sleep()

		# plot the error over time after finished
		self.plotter.plot_PID()

	def init_torque_mode(self):
		"""
		Switches Jaco to torque-control mode using ROS services
		"""
		# use service to set torque control parameters	
		service_address = prefix + '/in/set_torque_control_parameters'	
		rospy.wait_for_service(service_address)
		try:
			setTorqueParameters = rospy.ServiceProxy(service_address, SetTorqueControlParameters)
			setTorqueParameters()           
		except rospy.ServiceException, e:
			print "Service call failed: %s"%e
			return None	

		# use service to switch to torque control	
		service_address = prefix + '/in/set_torque_control_mode'	
		rospy.wait_for_service(service_address)
		try:
			switchTorquemode = rospy.ServiceProxy(service_address, SetTorqueControlMode)
			switchTorquemode(1)           
		except rospy.ServiceException, e:
			print "Service call failed: %s"%e
			return None	

	def torque_to_JointTorqueMsg(self, ):
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

	def PID_control(self, pos):
		"""
		Return a control torque based on PID control
		"""
		# deal with angle wraparound when computing difference
		error = -((self.target_pos - pos + math.pi)%(2*math.pi) - math.pi)
		
		return -self.controller.update_PID(error)


	def joint_torques_callback(self, msg):
		"""
		Reads the latest torque sensed by the robot and records it for 
		plotting & analysis
		"""
		# read the current joint torques from the robot
		torque_curr = np.array([msg.joint1,msg.joint2,msg.joint3,msg.joint4,msg.joint5,msg.joint6,msg.joint7]).reshape((7,1))

		# save running list of joint torques
		self.joint_torques = np.column_stack((self.joint_torques,torque_curr))

		# update the plot of joint torques over time
		self.plotter.update_joint_torque(torque_curr)

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
		self.torque = self.PID_control(pos_curr)

		# check if each angular torque is within set velocity limits
		for i in range(7):
			if self.torque[i][i] > self.max_torque[i][i]:
				self.torque[i][i] = self.max_torque[i][i]
			if self.torque[i][i] < -self.max_torque[i][i]:
				self.torque[i][i] = -self.max_torque[i][i]

		# update plotter with new error measurement and torque command
		cmd_tau = np.diag(self.controller.cmd).reshape((7,1))
		self.plotter.update_PID_plot(self.controller.p_error, self.controller.i_error, self.controller.d_error, cmd_tau)


if __name__ == '__main__':
	if len(sys.argv) < 11:
		print "ERROR: Not enough arguments. Specify p_gains, i_gains, d_gains, joint angles 1 - 7."
	else:	
		p_gains = float(sys.argv[1])
		i_gains = float(sys.argv[2])
		d_gains = float(sys.argv[3])

		j0 = float(sys.argv[4])
		j1 = float(sys.argv[5])
		j2 = float(sys.argv[6])
		j3 = float(sys.argv[7])
		j4 = float(sys.argv[8])
		j5 = float(sys.argv[9])
		j6 = float(sys.argv[10])

		PIDTorqueJaco(p_gains,i_gains,d_gains,j0,j1,j2,j3,j4,j5,j6)
		
	
