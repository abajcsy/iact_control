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
from geometry_msgs.msg import Vector3
import std_msgs.msg
import sensor_msgs.msg
from kinova_msgs.srv import *
from std_msgs.msg import Float32
from sympy import Point, Line

import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt


# 0 = impedance, 1 = deformation, 2 = learning w/impedance, 3 = learning w/gravcomp
MODE = 3


prefix = 'j2s7s300_driver'

pick_basic = [104.2,151.6,183.8,101.8,224.2,216.9,310.8]
place_higher = [233.0,132.4,200.5,137.8,248.7,243.2,264.8]

epsilon = 0.10
MAX_CMD_TORQUE = 40.0
INTERACTION_TORQUE_THRESHOLD = 8.0


class PIDVelJaco(object): 

	def __init__(self, ID, task, methodType, demo, record):
		"""
		Setup of the ROS node. Publishing computed torques happens at 100Hz.
		"""

		# start admittance control mode
		self.start_admittance_mode()

		# initialize trajectory
		start = np.array(pick_basic)*(math.pi/180.0)
		goal = np.array(place_higher)*(math.pi/180.0)
		self.start = start
		self.goal = goal
		self.start_time = 0.0
		self.final_time = 15.0
		self.step_time = 1.0
		self.weights = 0.0
		self.planner = trajopt_planner.Planner()
		self.planner.replan(self.start, self.goal, self.start_time, self.final_time, self.weights)
		self.planner.upsample(self.step_time)
		
		#setup the interaction mode
		self.interaction = False
		self.interaction_mode = False
		self.interaction_count = 0

		# save intermediate target position from degrees (default) to radians 
		self.target_pos = start.reshape((7,1))
		# save start configuration of arm
		self.start_pos = start.reshape((7,1))
		# save final goal configuration
		self.goal_pos = goal.reshape((7,1))

		# track if you have gotten to start/goal of path
		self.reached_start = False
		self.reached_goal = False

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
		self.P = 50.0*np.eye(7)
		self.I = 0.0*np.eye(7)
		self.D = 20.0*np.eye(7)
		self.controller = pid.PID(self.P,self.I,self.D,0,0)

		# ---- ROS Setup ---- #

		rospy.init_node("pid_trajopt")

		# create joint-velocity publisher
		self.torque_pub = rospy.Publisher(prefix + '/in/joint_torque', kinova_msgs.msg.JointTorque, queue_size=1)
		self.vel_pub = rospy.Publisher(prefix + '/in/joint_velocity', kinova_msgs.msg.JointVelocity, queue_size=1)

		# create subscriber to joint_torques
		rospy.Subscriber(prefix + '/out/joint_torques', kinova_msgs.msg.JointTorque, self.joint_torques_callback, queue_size=1)
		# create subscriber to joint_angles
		rospy.Subscriber(prefix + '/out/joint_angles', kinova_msgs.msg.JointAngles, self.joint_angles_callback, queue_size=1)

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
			print "starting admittance mode..."
			startForceControl()           
		except rospy.ServiceException, e:
			print "EXCEPTION WHEN STARTING admittance mode..."
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
		print "torque callback"
		torque_curr = np.array([msg.joint1,msg.joint2,msg.joint3,msg.joint4,msg.joint5,msg.joint6,msg.joint7]).reshape((7,1))

		if self.reached_start and not self.reached_goal:

			self.interaction = False
			self.interaction_count += 1
			for i in range(7):
				if i == 3:
					THRESHOLD = INTERACTION_TORQUE_THRESHOLD/2.0
				elif i > 3:
					THRESHOLD = INTERACTION_TORQUE_THRESHOLD/8.0
				else:
					THRESHOLD = INTERACTION_TORQUE_THRESHOLD
				if np.fabs(torque_curr[i][0]) > THRESHOLD:
					self.interaction = True
					self.interaction_count = 0
					print "interaction"

			if MODE > 0:
				if self.interaction:
					if MODE > 2 and self.interaction_mode == False:
						print "starting interaction mode"
						self.controller.set_gains(0.0*self.P,0.0*self.I,0.0*self.D, 0, 0)
						self.interaction_mode = True
					(waypts_deform, waypts_prev) = self.planner.deform(torque_curr, self.start_time)
					if MODE > 1 and waypts_deform != None:
						self.weights = self.planner.update(waypts_deform, waypts_prev)
						print "here is my weight: ", self.weights
						prev_start_time = self.start_time					
						self.planner.replan(self.start, self.goal, self.start_time, self.final_time, self.weights)
						if MODE > 2:						
							elapsed_time = time.time() - self.path_start_T - prev_start_time
						else:
							elapsed_time = 0.0
						self.planner.updateStart(self.start, elapsed_time)
						self.planner.upsample(self.step_time)
				elif MODE > 2 and self.interaction_mode and self.interaction_count > 2:
					print "ending interaction mode"
					self.controller.set_gains(self.P,self.I,self.D, 0, 0)
					self.interaction_mode = False
		

	def joint_angles_callback(self, msg):
		"""
		Reads the latest position of the robot and publishes an
		appropriate torque command to move the robot to the target
		"""
		print "angles callback"
		curr_pos = np.array([msg.joint1,msg.joint2,msg.joint3,msg.joint4,msg.joint5,msg.joint6,msg.joint7]).reshape((7,1))

		curr_pos = curr_pos*(math.pi/180.0)
		if self.reached_start and not self.reached_goal:
			self.start = curr_pos.reshape((1,7))
			self.start_time = time.time() - self.path_start_T

		self.update_target_pos(curr_pos)
		self.cmd = self.PID_control(curr_pos)

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
		if self.reached_start == False:

			dist_from_start = -((curr_pos - self.start_pos + math.pi)%(2*math.pi) - math.pi)			
			dist_from_start = np.fabs(dist_from_start)
			close_to_start = [dist_from_start[i] < epsilon for i in range(7)]
			is_at_start = all(close_to_start)

			if is_at_start:
				print "REACHED START!"
				self.reached_start = True
				self.path_start_T = time.time()
			else:
				self.target_pos = self.start_pos
		else:
			t = time.time() - self.path_start_T
			self.target_pos = self.planner.interpolate(t + 0.1)

			if self.reached_goal == False:
			
				dist_from_goal = -((curr_pos - self.goal_pos + math.pi)%(2*math.pi) - math.pi)			
				dist_from_goal = np.fabs(dist_from_goal)
				close_to_goal = [dist_from_goal[i] < epsilon for i in range(7)]
				is_at_goal = all(close_to_goal)
			
				if is_at_goal:
					self.reached_goal = True
			else:
				print "REACHED GOAL! Holding position at goal."



if __name__ == '__main__':
	if len(sys.argv) < 6:
		print "ERROR: Not enough arguments. Specify ID, task, methodType, demo, record"
	else:	
		ID = int(sys.argv[1])
		task = int(sys.argv[2])
		methodType = sys.argv[3]
		demo = sys.argv[4]
		record = sys.argv[5]

		PIDVelJaco(ID,task,methodType,demo,record)



"""
	def zeroJointTorqueMsg(self):
		jointCmd = kinova_msgs.msg.JointTorque()
		jointCmd.joint1 = 0.0;
		jointCmd.joint2 = 0.0;
		jointCmd.joint3 = 0.0;
		jointCmd.joint4 = 0.0;
		jointCmd.joint5 = 0.0;
		jointCmd.joint6 = 0.0;
		jointCmd.joint7 = 0.0;
	
		return jointCmd

	def init_torque_mode(self):
		# use service to switch to torque control	
		service_address = '/j2s7s300_driver/in/set_torque_control_mode'	
		rospy.wait_for_service(service_address)
		try:
			switchTorquemode = rospy.ServiceProxy(service_address, SetTorqueControlMode)
			switchTorquemode(1)           
			print "set torque mode!!"
		except rospy.ServiceException, e:
			print "Service call failed: %s"%e
			return None	

	def end_torque_mode(self):
		# switch back to position control after finished 
		service_address = '/j2s7s300_driver/in/set_torque_control_mode'	
		rospy.wait_for_service(service_address)
		try:
			switchTorquemode = rospy.ServiceProxy(service_address, SetTorqueControlMode)
			switchTorquemode(0)
			return None
		except rospy.ServiceException, e:
			print "Service call failed: %s"%e
			return None

	def set_admittance_params(self, reset=False):
		service_address = prefix+'/in/set_angular_force_control_params'
		rospy.wait_for_service(service_address)
		try:
			setAngularForceControlParams = rospy.ServiceProxy(service_address, SetAngularForceControlParams)
			if reset:
				setAngularForceControlParams()
			else:
				inertia = np.array([0.1,0.1,0.1,0.1,0.0,0.0,0.0])
				damping = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0])
				min_force = np.array([-10.0]*7)
				max_force = np.array([10.0]*7)
				setAngularForceControlParams(inertia,damping,min_force,max_force)         
				print "i set the angular force control params!"  

		except rospy.ServiceException, e:
			print "Service call failed: %s"%e
			return None	
"""	
