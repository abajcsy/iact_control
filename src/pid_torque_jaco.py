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
import planner

import kinova_msgs.msg
import geometry_msgs.msg
import std_msgs.msg
import sensor_msgs.msg
from kinova_msgs.srv import *
from std_msgs.msg import Float32
from sympy import Point, Line

import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt

prefix = 'j2s7s300_driver'

home_pos = [103.366,197.13,180.070,43.4309,265.11,257.271,287.9276]
candlestick_pos = [180.0]*7

pos1 = [14.30,162.95,190.75,124.03,176.10,188.25,167.94]
pos2 = [121.89,159.32,213.20,109.06,153.09,185.10,170.77]

waypt1 = [136.886, 200.805, 64.022, 116.637, 138.328, 122.469, 179.861]
waypt2 = [271.091, 225.708, 20.548, 158.572, 160.879, 183.520, 186.644]
waypt3 = [338.680, 172.142, 25.755, 96.798, 180.497, 137.340, 186.655]

traj = [waypt1, waypt2, waypt3]

epsilon = 0.08

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

	def __init__(self, p_gain, i_gain, d_gain):
		"""
		Setup of the ROS node. Publishing computed torques happens at 100Hz.
		"""

		# ---- ROS Setup ---- #
		rospy.init_node("pid_torque_jaco")

		# switch robot to torque-control mode
		self.init_torque_mode()

		# create joint-torque publisher
		self.torque_pub = rospy.Publisher(prefix + '/in/joint_torque', kinova_msgs.msg.JointTorque, queue_size=1)

		# create subscriber to joint_angles
		rospy.Subscriber(prefix + '/out/joint_angles', kinova_msgs.msg.JointAngles, self.joint_angles_callback, queue_size=1)
		# create subscriber to joint_torques
		rospy.Subscriber(prefix + '/out/joint_torques', kinova_msgs.msg.JointTorque, self.joint_torques_callback, queue_size=1)
		# create subscriber to joint_state
		rospy.Subscriber(prefix + '/out/joint_state', sensor_msgs.msg.JointState, self.joint_state_callback, queue_size=1)

		# ---- Trajectory Setup ---- #

		# list of waypoints along trajectory
		waypts = [None]*len(traj)
		for i in range(len(traj)):
			waypts[i] = np.array(traj[i]).reshape((7,1))*(math.pi/180.0)

		# scaling on speed
		self.alpha = 1.0

		# time for each (linear) segment of trajectory
		deltaT = [2.0, 2.0] 
		# blending time (sec)
		t_b = 1.0

		# get trajectory planner
		self.planner = planner.PathPlanner(waypts, t_b, deltaT, self.alpha)

		# sets max execution time 
		self.T = self.planner.get_t_f()

		# save intermediate target position from degrees (default) to radians 
		self.target_pos = waypts[0]
		# save start configuration of arm
		self.start_pos = waypts[0]
		# save final goal configuration
		self.goal_pos = waypts[-1]
	
		# track if you have gotten to start/goal of path
		self.reached_start = False
		self.reached_goal = False

		# TODO THIS IS EXPERIMENTAL
		self.interaction = False

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

		# keeps running time since beginning of program execution
		self.process_start_T = time.time() 
		# keeps running time since beginning of path
		self.path_start_T = None 

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
		tot_path_time = time.time() - self.path_start_T
		self.plotter.plot_PID(tot_path_time)

		# switch back to position control after finished 
		service_address = prefix + '/in/set_torque_control_mode'	
		rospy.wait_for_service(service_address)
		try:
			switchTorquemode = rospy.ServiceProxy(service_address, SetTorqueControlMode)
			switchTorquemode(0)
			return None
		except rospy.ServiceException, e:
			print "Service call failed: %s"%e
			#return None

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
			#return None	

		# use service to switch to torque control	
		service_address = prefix + '/in/set_torque_control_mode'	
		rospy.wait_for_service(service_address)
		try:
			switchTorquemode = rospy.ServiceProxy(service_address, SetTorqueControlMode)
			switchTorquemode(1)           
		except rospy.ServiceException, e:
			print "Service call failed: %s"%e
			return None	

	def torque_to_JointTorqueMsg(self):
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

		print "torque_curr: " + str(torque_curr)
		for i in range(7):
			if np.fabs(torque_curr[i][0]) > 7:
				print "I HAVE SET THE INTERACTION"
				self.interaction = True
				break
			#else: 
				#self.interaction = False

		# update the plot of joint torques over time
		t = time.time() - self.process_start_T
		self.plotter.update_joint_torque(torque_curr, t)

	def joint_state_callback(self, msg):
		"""
		Reads the latest vel sensed by the robot and records it for 
		plotting & analysis
		"""
		j_vel = np.array([msg.velocity[0],msg.velocity[1],msg.velocity[2],msg.velocity[3],msg.velocity[4],msg.velocity[5],msg.velocity[6]]).reshape((7,1))

		# update the plot of joint torques over time
		t = time.time() - self.process_start_T
		self.plotter.update_joint_vel(j_vel, t)

	def joint_angles_callback(self, msg):
		"""
		Reads the latest position of the robot and publishes an
		appropriate torque command to move the robot to the target
		"""
		# read the current joint angles from the robot
		curr_pos = np.array([msg.joint1,msg.joint2,msg.joint3,msg.joint4,msg.joint5,msg.joint6,msg.joint7]).reshape((7,1))

		# convert to radians
		curr_pos = curr_pos*(math.pi/180.0)
		
		# update target position to move to depending on:
		# - if moving to START of desired trajectory or 
		# - if moving ALONG desired trajectory
		self.update_target_pos(curr_pos)

		# update torque from PID based on current position
		self.torque = self.PID_control(curr_pos)

		# check if each angular torque is within set limits
		for i in range(7):
			if self.torque[i][i] > self.max_torque[i][i]:
				self.torque[i][i] = self.max_torque[i][i]
			if self.torque[i][i] < -self.max_torque[i][i]:
				self.torque[i][i] = -self.max_torque[i][i]

		# update plotter with new error measurement, torque command, and path time
		curr_time = time.time() - self.process_start_T
		cmd_tau = np.diag(self.controller.cmd).reshape((7,1))

		#print "target_pos: " + str(self.target_pos)
		#print "curr_pos: " + str(curr_pos)
		print "dist to target: " + str(self.target_pos - curr_pos)

		self.plotter.update_PID_plot(self.controller.p_error, self.controller.i_error, self.controller.d_error, cmd_tau, curr_time)

	def update_target_pos(self, curr_pos):
		"""
		Takes the current position of the robot. Determines what the next
		target position to move to should be depending on:
		- if robot is moving to start of desired trajectory or 
		- if robot is moving along the desired trajectory 
		"""
		# check if the arm is at the start of the path to execute
		if not self.reached_start:
			dist_from_start = np.fabs(curr_pos - self.start_pos)

			# check if every joint is close enough to start configuration
			close_to_start = [dist_from_start[i] < epsilon for i in range(7)]

			# if all joints are close enough, robot is at start
			is_at_start = all(close_to_start)

			if is_at_start:
				self.reached_start = True
				self.path_start_T = time.time()
				# for plotting, save time when path execution started
				self.plotter.set_path_start_time(time.time() - self.process_start_T)
			else:
				print "NOT AT START"
				# if not at start of trajectory yet, set starting position 
				# of the trajectory as the current target position
				self.target_pos = self.start_pos
		else:
			print "REACHED START --> EXECUTING PATH"
			t = time.time() - self.path_start_T
			(self.T, self.target_pos) = self.planner.linear_path(t, curr_pos)
			#(self.T, self.target_pos) = self.planner.time_trajectory(t)
			print "t: " + str(t)
			print "T: " + str(self.T)

		# check if the arm reached the goal, and restart path
		if not self.reached_goal:
			dist_from_goal = np.fabs(curr_pos - self.goal_pos)

			# check if every joint is close enough to goal configuration
			close_to_goal = [dist_from_goal[i] < epsilon for i in range(7)]
			
			# if all joints are close enough, robot is at goal
			is_at_goal = all(close_to_goal)
			
			if is_at_goal:
				self.reached_goal = True
		else:
			print "REACHED GOAL --> RE-EXECUTING PATH"

			#print "OLD start: " + str(self.start_pos)
			#print "OLD goal: " + str(self.goal_pos)

			old_start = np.copy(self.start_pos)
			self.target_pos = self.goal_pos
			# save start configuration of arm
			self.start_pos = self.goal_pos
			# save final goal configuration
			self.goal_pos = old_start

			#print "NEW start: " + str(self.start_pos)
			#print "NEW goal: " + str(self.goal_pos)

			# update planner
			# TODO WARNING!: THIS IS NOT FIXED FOR > 2 WAYPOINTS. 
			# TODO Need to reverse order of waypts in trajectory to replay from start

			# set new start and goal positions in the planner
			self.planner.s = self.start_pos
			self.planner.g = self.goal_pos

			# update new path start time
			self.path_start_T = time.time()

			# compute new total time for path to execute
			self.planner.t_f = self.alpha*(linalg.norm(self.start_pos-self.goal_pos)**2)
			self.planner.total_t = self.planner.total_t + self.alpha*(linalg.norm(self.start_pos-self.goal_pos)**2)

			self.reached_goal = False

if __name__ == '__main__':
	if len(sys.argv) < 6:
		print "ERROR: Not enough arguments. Specify p_gains, i_gains, d_gains, start, goal."
	else:	
		p_gains = float(sys.argv[1])
		i_gains = float(sys.argv[2])
		d_gains = float(sys.argv[3])

		#start = str(sys.argv[4])
		#goal = str(sys.argv[5])

		PIDTorqueJaco(p_gains,i_gains,d_gains)
	
