#! /usr/bin/env python
"""
This node demonstrates velocity-based PID control by moving the Jaco
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
import trajopt_planner
import traj
import ros_utils
import experiment_utils

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

pick = [104.2, 151.6, 183.8, 101.8, 224.2, 216.9, 220.8]
place = [210.8, 101.6, 192.0, 114.7, 222.2, 246.1, 252.0]

epsilon = 0.10
MAX_CMD_TORQUE = 40.0
INTERACTION_TORQUE_THRESHOLD = 8.0

TABLE_TASK = 0
LAPTOP_TASK = 1
COFFEE_TASK = 2

IMPEDANCE = 'A'
LEARNING = 'B'

class PIDVelJaco(object): 
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
		/j2s7s300_driver/out/joint_angles	- Jaco sensed joint angles
		/j2s7s300_driver/out/joint_torques	- Jaco sensed joint torques
	
	Publishes to:
		/j2s7s300_driver/in/joint_velocity	- Jaco commanded joint velocities 
	
	Required parameters:
		p_gain, i_gain, d_gain    - gain terms for the PID controller
		sim_flag 				  - flag for if in simulation or not
	"""

	def __init__(self, p_gain, i_gain, d_gain, task, methodType):
		"""
		Setup of the ROS node. Publishing computed torques happens at 100Hz.
		"""

		# start admittance control mode
		self.start_admittance_mode()

		# ---- Trajectory Setup ---- #

		# total time for trajectory
		self.T = 15.0

		# initialize trajectory weights
		self.weights = 0

		start = np.array(pick)*(math.pi/180.0)
		goal = np.array(place)*(math.pi/180.0)
		self.start = start
		self.goal = goal

		# task type - table, laptop, or coffee task
		self.task = task

		# method type - A=IMPEDANCE, B=LEARNING
		self.methodType = methodType

		# create the trajopt planner and plan from start to goal
		self.planner = trajopt_planner.Planner(self.task)
		self.planner.replan(self.start, self.goal, self.weights, 0.0, self.T, 0.5)

		# save intermediate target position from degrees (default) to radians 
		self.target_pos = start.reshape((7,1))
		# save start configuration of arm
		self.start_pos = start.reshape((7,1))
		# save final goal configuration
		self.goal_pos = goal.reshape((7,1))

		# track if you have gotten to start/goal of path
		self.reached_start = False
		self.reached_goal = False

		# keeps running time since beginning of program execution
		self.process_start_T = time.time() 
		# keeps running time since beginning of path
		self.path_start_T = None 

		# ----- Controller Setup ----- #

		# stores maximum COMMANDED joint torques		
		self.max_cmd = MAX_CMD_TORQUE*np.eye(7)
		# stores current COMMANDED joint torques
		self.cmd = np.eye(7) 
		# stores current joint MEASURED joint torques
		self.joint_torques = np.zeros((7,1))

		# P, I, D gains 
		self.P = p_gain*np.eye(7)
		self.I = i_gain*np.eye(7)
		self.D = d_gain*np.eye(7)
		self.controller = pid.PID(self.P,self.I,self.D,0,0)

		# ---- Experimental Utils ---- #

		self.expUtil = experiment_utils.ExperimentUtils()

		# ---- ROS Setup ---- #

		rospy.init_node("pid_trajopt")

		# create joint-velocity publisher
		self.vel_pub = rospy.Publisher(prefix + '/in/joint_velocity', kinova_msgs.msg.JointVelocity, queue_size=1)

		# create subscriber to joint_angles
		rospy.Subscriber(prefix + '/out/joint_angles', kinova_msgs.msg.JointAngles, self.joint_angles_callback, queue_size=1)
		# create subscriber to joint_torques
		rospy.Subscriber(prefix + '/out/joint_torques', kinova_msgs.msg.JointTorque, self.joint_torques_callback, queue_size=1)

		# publish to ROS at 100hz
		r = rospy.Rate(100) 

		print "----------------------------------"
		print "Moving robot, press ENTER to quit:"
		
		while not rospy.is_shutdown():

			if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
				line = raw_input()
				break

			self.vel_pub.publish(ros_utils.cmd_to_JointVelocityMsg(self.cmd))
			r.sleep()
		
		print "----------------------------------"

		# save and plot experimental data
		#print "Saving experimental data to file..."
		#forcefilename = "force" + str(ID) + str(task) + methodType + ".csv"		
		#self.expUtil.save_tauH(filename)	

		# end admittance control mode
		self.stop_admittance_mode()

	def start_admittance_mode(self):
		"""
		Switches Jaco to admittance-control mode using ROS services
		"""
		service_address = prefix+'/in/start_force_control'	
		rospy.wait_for_service(service_address)
		try:
			startForceControl = rospy.ServiceProxy(service_address, Start)
			startForceControl()           
		except rospy.ServiceException, e:
			print "Service call failed: %s"%e
			return None	

	def stop_admittance_mode(self):
		"""
		Switches Jaco to position-control mode using ROS services
		"""
		service_address = prefix+'/in/stop_force_control'	
		rospy.wait_for_service(service_address)
		try:
			stopForceControl = rospy.ServiceProxy(service_address, Stop)
			stopForceControl()           
		except rospy.ServiceException, e:
			print "Service call failed: %s"%e
			return None	

	def PID_control(self, pos):
		"""
		Return a control torque based on PID control
		"""
		error = -((self.target_pos - pos + math.pi)%(2*math.pi) - math.pi)
		return -self.controller.update_PID(error)

	def joint_torques_callback(self, msg):
		"""
		Reads the latest torque sensed by the robot and records it for 
		plotting & analysis
		"""
		# read the current joint torques from the robot
		torque_curr = np.array([msg.joint1,msg.joint2,msg.joint3,msg.joint4,msg.joint5,msg.joint6,msg.joint7]).reshape((7,1))

		#print "Current torque: " + str(torque_curr)
		interaction = False
		for i in range(7):
			THRESHOLD = INTERACTION_TORQUE_THRESHOLD
			if self.reached_start and i >= 3:
				THRESHOLD = 1.0
			if np.fabs(torque_curr[i][0]) > THRESHOLD:
				interaction = True
			else:
				# zero out torques below threshold for cleanliness
				torque_curr[i][0] = 0.0
		#print "Cleaned torque: " + str(torque_curr)

		# if experienced large enough interaction force, then deform traj
		if interaction:
			print "--- INTERACTION ---"
			if self.reached_start and not self.reached_goal:
				timestamp = time.time() - self.path_start_T
				self.expUtil.update_tauH(timestamp, torque_curr)

			if self.methodType == LEARNING:
				self.weights = self.planner.learnWeights(torque_curr)
				self.planner.replan(self.start, self.goal, self.weights, 0.0, self.T, 0.5)

	def joint_angles_callback(self, msg):
		"""
		Reads the latest position of the robot and publishes an
		appropriate torque command to move the robot to the target
		"""
		# read the current joint angles from the robot
		curr_pos = np.array([msg.joint1,msg.joint2,msg.joint3,msg.joint4,msg.joint5,msg.joint6,msg.joint7]).reshape((7,1))

		# convert to radians
		curr_pos = curr_pos*(math.pi/180.0)	

		# update the OpenRAVE simulation 
		#self.planner.update_curr_pos(curr_pos)

		# update target position to move to depending on:
		# - if moving to START of desired trajectory or 
		# - if moving ALONG desired trajectory
		self.update_target_pos(curr_pos)

		# update the experiment utils executed trajectory tracker
		if self.reached_start and not self.reached_goal:
			timestamp = time.time() - self.path_start_T
			self.expUtil.update_traj(timestamp, curr_pos)

		# update cmd from PID based on current position
		self.cmd = self.PID_control(curr_pos)

		# check if each angular torque is within set limits
		for i in range(7):
			if self.cmd[i][i] > self.max_cmd[i][i]:
				self.cmd[i][i] = self.max_cmd[i][i]
			if self.cmd[i][i] < -self.max_cmd[i][i]:
				self.cmd[i][i] = -self.max_cmd[i][i]

	def update_target_pos(self, curr_pos):
		"""
		Takes the current position of the robot. Determines what the next
		target position to move to should be depending on:
		- if robot is moving to start of desired trajectory or 
		- if robot is moving along the desired trajectory 
		"""
		# check if the arm is at the start of the path to execute
		if not self.reached_start:

			dist_from_start = -((curr_pos - self.start_pos + math.pi)%(2*math.pi) - math.pi)			
			dist_from_start = np.fabs(dist_from_start)

			# check if every joint is close enough to start configuration
			close_to_start = [dist_from_start[i] < epsilon for i in range(7)]

			# if all joints are close enough, robot is at start
			is_at_start = all(close_to_start)

			if is_at_start:
				self.reached_start = True
				self.path_start_T = time.time()
				self.expUtil.set_startT(self.path_start_T)
			else:
				print "NOT AT START"
				# if not at start of trajectory yet, set starting position 
				# of the trajectory as the current target position
				self.target_pos = self.start_pos
		else:
			print "REACHED START --> EXECUTING PATH"

			t = time.time() - self.path_start_T
			print "t: " + str(t)

			# get next target position from position along trajectory
			self.target_pos = self.planner.interpolate(t)

		# check if the arm reached the goal, and restart path
		if not self.reached_goal:
			
			dist_from_goal = -((curr_pos - self.goal_pos + math.pi)%(2*math.pi) - math.pi)			
			dist_from_goal = np.fabs(dist_from_goal)

			# check if every joint is close enough to goal configuration
			close_to_goal = [dist_from_goal[i] < epsilon for i in range(7)]
			
			# if all joints are close enough, robot is at goal
			is_at_goal = all(close_to_goal)
			
			if is_at_goal:
				self.reached_goal = True
		else:
			print "REACHED GOAL! Holding position at goal."
			self.target_pos = self.goal_pos
			# TODO: this should only set it once!
			self.expUtil.set_endT(time.time())

if __name__ == '__main__':
	if len(sys.argv) < 6:
		print "ERROR: Not enough arguments. Specify p_gains, i_gains, d_gains, task, methodType."
	else:	
		p_gains = float(sys.argv[1])
		i_gains = float(sys.argv[2])
		d_gains = float(sys.argv[3])
		task = int(sys.argv[4])
		methodType = sys.argv[5]

		PIDVelJaco(p_gains,i_gains,d_gains,task,methodType)
	
