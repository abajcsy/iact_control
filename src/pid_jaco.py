#! /usr/bin/env python
"""
This node demonstrates PID control by moving the Jaco
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
#from robot_control_modules import *

import numpy as np
import matplotlib.pyplot as plt

prefix = 'j2s7s300_driver'

class PIDJaco(object): 
	"""
	This class represents a node that moves the Jaco with PID control.
	The joint velocities are computed as:
		
		V = -K_p(e) - K_d(e_dot) - K_i*Integral(e)

	where:

		e = (target_joint configuration) - (current joint configuration)
		e_dot = derivative of error
		K_p = accounts for present values of position error
		K_i = accounts for past values of error, accumulates error over time
		K_d = accounts for possible future trends of error, based on current rate of change
	
	Subscribes to: 
		/j2s7s300_driver/out/joint_angles	- Jaco joint angles
	
	Publishes to:
		/j2s7s300_driver/in/joint_velocity	- Jaco joint velocities 
	
	Required parameters:
		p_gain, i_gain, d_gain    - gain terms for the PID controller
		j0, ... , j6				  - goal configuration for joints 1-7 (in degrees)
	"""

	def __init__(self, p_gain, i_gain, d_gain, j0,j1,j2,j3,j4,j5,j6):
		"""
		Setup of the ROS node. Publishing computed velocities happens at 100Hz.
		"""

		rospy.init_node("pid_jaco")

		# create angular-velocity publisher
		self.velocity_pub = rospy.Publisher(prefix + '/in/joint_velocity', kinova_msgs.msg.JointVelocity, queue_size=1)

		# create subscriber to joint_angles
		rospy.Subscriber(prefix + '/out/joint_angles', kinova_msgs.msg.JointAngles, self.joint_angles_callback, queue_size=1)

		# convert target position from degrees (default) to radians 		
		self.target_pos = np.array([j0,j1,j2,j3,j4,j5,j6]).reshape((7,1))* (math.pi/180.0)

		self.max_velocity = 100*np.eye(7)
		self.velocity = np.eye(7) 

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

		#print "-----------------------------"
		#print 'Moving robot, press q to quit:'
		#nb = ''

		# publish to ROS at 100hz
		r = rospy.Rate(100) 
		while not rospy.is_shutdown(): #and nb != 'q':
			#nb = getch.getche() # need to import 
			self.velocity_pub.publish(self.velocity_to_JointVelocityMsg()) 
			r.sleep()

		# plot the error over time after finished
		self.plotter.plot_PID()

	def velocity_to_JointVelocityMsg(self):
		"""
		Returns a JointVelocity Kinova msg from an array of torques
		"""
		jointCmd = kinova_msgs.msg.JointVelocity()
		jointCmd.joint1 = self.velocity[0][0];
		jointCmd.joint2 = self.velocity[1][1];
		jointCmd.joint3 = self.velocity[2][2];
		jointCmd.joint4 = self.velocity[3][3];
		jointCmd.joint5 = self.velocity[4][4];
		jointCmd.joint6 = self.velocity[5][5];
		jointCmd.joint7 = self.velocity[6][6];

		return jointCmd

	def PID_control(self, pos):
		"""
		Return a control torque based on PID control
		"""
		# deal with angle wraparound when computing difference
		error = -((self.target_pos - pos + math.pi)%(2*math.pi) - math.pi)
		
		return -self.controller.update_PID(error)

	def joint_angles_callback(self, msg):
		"""
		Reads the latest position of the robot and publishes an
		appropriate torque command to move the robot to the target
		"""
		# read the current joint angles from the robot
		pos_curr = np.array([msg.joint1,msg.joint2,msg.joint3,msg.joint4,msg.joint5,msg.joint6,msg.joint7]).reshape((7,1))

		# convert to radians
		pos_curr = pos_curr*(math.pi/180.0)

		# update velocity from PID based on current position
		self.velocity = self.PID_control(pos_curr)

		# check if each angular velocity is within set velocity limits
		for i in range(7):
			if self.velocity[i][i] > self.max_velocity[i][i]:
				self.velocity[i][i] = self.max_velocity[i][i]
			if self.velocity[i][i] < -self.max_velocity[i][i]:
				self.velocity[i][i] = -self.max_velocity[i][i]


		# update plotter with new error measurement 
		self.plotter.update_PID_plot(self.controller.p_error, self.controller.i_error, self.controller.d_error)


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

		PIDJaco(p_gains,i_gains,d_gains,j0,j1,j2,j3,j4,j5,j6)
		
	
