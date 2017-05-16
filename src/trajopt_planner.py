import numpy as np
from numpy import linalg
from numpy import linspace
import matplotlib.pyplot as plt
import time
import math

from sympy import symbols
from sympy import lambdify

import trajoptpy
import or_trajopt
import openravepy
from openravepy import *

import interactpy
from interactpy import *
from util import *

import logging
import pid

import sim_robot

logging.getLogger('prpy.planning.base').addHandler(logging.NullHandler())

class Planner(object):
	"""
	This class represents plans a trajectory from start to goal 
	with TrajOpt. 

	Required parameters:
		start    - start cofiguration (at t = 0)
		goal	 - goal configuration (at t = T)
		T 		 - total time for trajectory execution
	"""

	def __init__(self, start, goal, T, waypts=None):	
		# set first and last waypt as start and goal
		# NOTE: assume that start and goal are in radians

		# TODO: will have to convert convert from 7x1 np matrix to 1x10 np matrix 
		# need to add padding to start
		padding = np.array([0,0,0])
		start = np.append(start, padding, 1)
		self.s = start
		self.g = goal
		self.totalT = T
		self.curr_pos = self.s

		print "totalT: " + str(self.totalT)

		# initialize openrave and compute waypts
		model_filename = 'jaco_dynamics'
		#model_filename = 'jaco'
		self.env, self.robot = interact.initialize_empty(model_filename, empty=True)
		physics = RaveCreatePhysicsEngine(self.env,'ode')
		self.env.SetPhysicsEngine(physics)

		# insert any objects you want into environment

		# bodies for visualization in RVIZ
		self.bodies = []
		coords = [0.5, 0.3, 0.8]
		self.plotPoint(coords)

		viewer = self.env.GetViewer()

		viewer.SetSize(1000,1000)
		viewer.SetCamera([[ 0.94684722, -0.12076704,  0.29815376,  0.21004671],
			   [-0.3208323 , -0.42191214,  0.84797216, -1.40675116],
			   [ 0.0233876 , -0.89855744, -0.43823231,  1.04685986],
			   [ 0.        ,  0.        ,  0.        ,  1.        ]])

		viewer.SetBkgndColor([0.8,0.8,0.8])

		# plan trajectory from self.s to self.g with self.totalT time
		if waypts is None:
			self.replan(self.s, self.totalT)
			print "Waypoint times T:" + str(self.wayptsT) 
		else:
			self.set_waypts(waypts, self.totalT)

		# store cartesian waypoints for plotting and debugging
		self.cartesian_waypts = self.get_cartesian_waypts()
		self.sampled_cartesian_waypts = self.plotTraj()
		#self.plot_cartesian_waypts(self.cartesian_waypts)


	def plotPoint(self, coords):
		with self.env:
			body = RaveCreateKinBody(self.env, '')
			body.InitFromBoxes(np.array([[coords[0], coords[1], coords[2],
						  0.1, 0.1, 0.1]]))
			body.SetName(str(len(self.bodies)))
			self.env.Add(body, True)
			self.bodies.append(body)

	def replan(self, newStart, T):
		"""
		Computes a plan from newStart to self.g taking T total time.
		"""
		self.s = newStart
		self.totalT = T

		print "in replan...totalT: " + str(self.totalT)
		self.robot.SetDOFValues(self.s)
		orig_ee = self.robot.arm.hand.GetTransform()
		
		self.waypts = self.robot.arm.PlanToConfiguration(self.g)
		self.num_waypts = self.waypts.GetNumWaypoints()

		print "selfs: " + str(self.s[:7])
		print "selfg: " + str(self.g)
		if np.array_equal(self.s[:7], self.g):
			print "START AND GOAL ARE THE SAME. Just holding position."

		print "in replan...num_waypts: " + str(self.num_waypts)
		for i in range(self.num_waypts):
			print self.waypts.GetWaypoint(i)

		# compute time, T_i for each of the waypoints i in {1,...n}
		self.wayptsT = [None]*self.num_waypts
		T_sum = 0.0
		if self.num_waypts >= 2:
			for i in range(self.num_waypts):
				self.wayptsT[i] = T_sum
				print "in replan...T_sum: " + str(T_sum)
				T_sum += self.totalT/(self.num_waypts-1)
		else:
			self.wayptsT[0] = self.totalT

	def set_waypts(self, waypoints, T):
		"""
		Sets the waypoints for trajectory taking T time. Input is in degrees
		"""
		self.totalT = T

		print "in set_waypts...totalT: " + str(self.totalT)
		self.robot.SetDOFValues(self.s)
		orig_ee = self.robot.arm.hand.GetTransform()
		
		self.waypts = []
		for i in range(len(waypoints)):
			self.waypts.append(waypoints[i]*(math.pi/180.0))
		self.num_waypts = len(waypoints)

		# compute time, T_i for each of the waypoints i in {1,...n}
		self.wayptsT = [None]*self.num_waypts
		T_sum = 0.0
		if self.num_waypts >= 2:
			for i in range(self.num_waypts):
				self.wayptsT[i] = T_sum
				print "in set_waypts...T_sum: " + str(T_sum)
				T_sum += self.totalT/(self.num_waypts-1)
		else:
			self.wayptsT[0] = self.totalT

	def get_cartesian_waypts(self):
		"""
		Returns list of waypoints along trajectory in task-space
		- Return type: list of length 3 numpy arrays
		"""
		cartesian = []
		for i in range(self.num_waypts):
			waypoint = self.waypts.GetWaypoint(i)
			dof = np.append(waypoint, np.array([1,1,1]))
			tf = transformToCartesian(dofToTransform(self.robot, dof))
			print "cartesian " + str(i) + ": "  + str(cartesian)
			cartesian.append(tf)
		print "cartesian: " + str(cartesian)
		return np.array(cartesian)

	def plot_cartesian_waypts(self, cartesian):
		"""
		Plots cartesian waypoints in OpenRAVE
		"""
		for i in range(self.num_waypts):
			plotPoint(self.env, self.bodies, cartesian[i])

	def get_robot(self):
		"""
		Returns robot model
		"""
		return self.robot

	def execute_path_sim(self):
		"""
		Executes in simulation the planned trajectory
		"""
		self.robot.ExecutePath(self.waypts)

	def update_curr_pos(self, curr_pos):
		"""
		Updates DOF values based on curr_pos
		"""
		pos = np.array([curr_pos[0][0],curr_pos[1][0],curr_pos[2][0],curr_pos[3][0],curr_pos[4][0],curr_pos[5][0],curr_pos[6][0],0,0,0])
		self.curr_pos = pos

		self.robot.SetDOFValues(self.curr_pos)

	def interpolate(self, t):
		"""
		Gets the next desired position along trajectory
		by interpolating between waypoints given the current t.
		"""
		target_pos = self.waypts.GetWaypoint(0)
		print "NUMBER WAYPOINTS: " + str(self.num_waypts)
		# TODO CHECK CORNER CASES START/END	
		if self.num_waypts >= 2:
			for i in range(self.num_waypts-1):
				# if between two waypoints, interpolate
				if t > self.wayptsT[i] and t < self.wayptsT[i+1]:
					print "between waypt " + str(i) + " and " + str(i+1)
					# linearly interpolate between waypts
					prev = self.waypts.GetWaypoint(i)
					next = self.waypts.GetWaypoint(i+1)
					Tprev = self.wayptsT[i]
					Tnext = self.wayptsT[i+1]
					deltaT = Tnext - Tprev
					theta = (next - prev)*(1/deltaT)*t + prev
					target_pos = theta
				# if exactly at a waypoint, return that waypoint
				elif t == self.wayptsT[i]:
					target_pos = self.waypts.GetWaypoint(i)			
		else:
			print "ONLY ONE WAYPT, CAN'T INTERPOLATE."
	
		# if times up, just go to goal
		if t > self.totalT:
			print "TIME IS UP. GOING TO FINAL WAYPOINT."
			target_pos = self.waypts.GetWaypoint(self.num_waypts-1)	
		target_pos = np.array(target_pos).reshape((7,1))
		return target_pos

	def interpolate2(self, t):
		"""
		Gets the next desired position along trajectory
		by interpolating between waypoints given the current t.
		"""
		target_pos = self.waypts[0]
		# TODO CHECK CORNER CASES START/END	
		if self.num_waypts >= 2:
			for i in range(self.num_waypts-1):
				# if between two waypoints, interpolate
				if t > self.wayptsT[i] and t < self.wayptsT[i+1]:
					# linearly interpolate between waypts
					prev = self.waypts[i]
					next = self.waypts[i+1]
					Tprev = self.wayptsT[i]
					Tnext = self.wayptsT[i+1]
					deltaT = Tnext - Tprev
					theta = (next - prev)*(1/deltaT)*t + prev
					target_pos = theta
				# if exactly at a waypoint, return that waypoint
				elif t == self.wayptsT[i]:
					target_pos = self.waypts[i]		
		else:
			print "ONLY ONE WAYPT, CAN'T INTERPOLATE."
	
		# if times up, just go to goal
		if t > self.totalT:
			print "TIME IS UP. GOING TO FINAL WAYPOINT."
			target_pos = self.waypts[self.num_waypts-1]	
		target_pos = np.array(target_pos).reshape((7,1))
		return target_pos

	def sampleWaypoints(self, traj):
		"""
		Samples waypoints every 0.5 seconds along the trajectory
		Parameters
		----------
		traj : OpenRAVE trajectory
		Returns
		-------
		2D-array of DOFs
		"""

		if traj.GetDuration() == 0:
			print "Retiming trajectory..."
			planningutils.RetimeTrajectory(traj)
		duration = traj.GetDuration()
		dofs = traj.SamplePoints2D(np.append(np.arange(0, duration, 0.5), duration))

		# for some reason, rrt creates dofs of dimension 15????
		dofs = dofs[:, :7]

		return dofs

	def sampleWaypts2(self, traj):
		t = 0.0
		dofs = []
		while t < self.totalT:
			theta = self.interpolate(t)
			dofs.append(theta)
			t += 0.1
		print "dofs: " + str(dofs)
		return dofs

	def plotTraj(self):
		"""
		Plots the best trajectory found or planned
		"""
		waypoints = self.sampleWaypts2(self.waypts) 
		#waypoints = self.sampleWaypoints(self.waypts)

		self.bodies += plotWaypoints(self.env, self.robot, waypoints)
		return waypoints

if __name__ == '__main__':
	
	home = np.array([103.366,197.13,180.070,43.4309,265.11,257.271,287.9276])
	goal = np.array([338.680, 172.142, 25.755, 96.798, 180.497, 137.340, 186.655])
	candlestick = np.array([180.0]*7)	
	
	waypt1 = np.array([136.886, 200.805, 64.022, 116.637, 138.328, 122.469, 179.861])
	waypt2 = np.array([271.091, 225.708, 20.548, 158.572, 160.879, 183.520, 186.644])
	waypt3 = np.array([338.680, 172.142, 25.755, 96.798, 180.497, 137.340, 186.655])

	home_pos = home
	goal_pos = goal

	s = np.array(home_pos)*(math.pi/180.0)
	g = np.array(candlestick)*(math.pi/180.0)

	#robot_starting_dofs = np.array([-1, 2, 0, 2, 0, 4, 0])
	#goal = np.array([0,  2.9 ,  0.0 ,  2.1 ,  0. ,  4. ,  0.])

	#s = robot_starting_dofs
	#g = goal

	print s
	print g

	T = 8.0
	trajplanner = Planner(s,g,T)
	#t = 0.8
	#theta = trajplanner.interpolate(t)
	trajplanner.execute_path_sim()
	time.sleep(20)
