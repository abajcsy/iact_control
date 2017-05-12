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

	def __init__(self, start, goal, T):	
		# set first and last waypt as start and goal
		# NOTE: assume that start and goal are in radians

		# TODO: will have to convert convert from 7x1 np matrix to 1x10 np matrix 
		# need to add padding to start
		padding = np.array([0,0,0])
		start = np.append(start, padding, 1)
		self.s = start
		self.g = goal
		self.totalT = T

		print "totalT: " + str(self.totalT)

		# initialize openrave and compute waypts
		#model_filename = 'jaco_original'
		self.env, self.robot = interact.initialize_empty()

		viewer = self.env.GetViewer()
		viewer.SetCamera([[ 0.94684722, -0.12076704,  0.29815376,  0.21004671],
		   [-0.3208323 , -0.42191214,  0.84797216, -1.40675116],
		   [ 0.0233876 , -0.89855744, -0.43823231,  1.04685986],
		   [ 0.        ,  0.        ,  0.        ,  1.        ]])

		# plan trajectory from self.s to self.g with self.totalT time
		self.replan(self.s, self.totalT)

		print "Waypoint times T:" + str(self.wayptsT) 

	def execute_path_sim(self):
		"""
		Executes in simulation the planned trajectory
		"""
		self.robot.ExecutePath(self.waypts)

	def replan(self, newStart, T):
		"""
		Computes a plan from newStart to self.g taking T time.
		"""
		self.s = newStart
		self.totalT = T

		print "in replan...totalT: " + str(self.totalT)
		self.robot.SetDOFValues(self.s)
		orig_ee = self.robot.arm.hand.GetTransform()
		
		self.waypts = self.robot.arm.PlanToConfiguration(self.g)
		self.num_waypts = self.waypts.GetNumWaypoints()
		print "in replan...num_waypts: " + str(self.num_waypts)
		for i in range(self.num_waypts):
			print self.waypts.GetWaypoint(i)

		# compute time, T_i for each of the waypoints i in {1,...n}
		self.wayptsT = [None]*self.num_waypts
		T_sum = 0.0
		for i in range(self.num_waypts):
			self.wayptsT[i] = T_sum
			print "in replan...T_sum: " + str(T_sum)
			T_sum += self.totalT/(self.num_waypts-1)

	def interpolate(self, t):
		"""
		Gets the next desired position along trajectory
		by interpolating between waypoints given the current t.
		"""
		target_pos = self.waypts.GetWaypoint(0)
		# TODO CHECK CORNER CASES START/END	
		if self.num_waypts >= 2:
			for i in range(self.num_waypts-1):
				# if between two waypoints, interpolate
				if t > self.wayptsT[i] and t < self.wayptsT[i+1]:
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
			planningutils.RetimeTrajectory(traj)
		duration = traj.GetDuration()
		dofs = traj.SamplePoints2D(np.append(np.arange(0, duration, 0.5), duration))
		#print "duration: " + str(duration)

		# for some reason, rrt creates dofs of dimension 15????
		dofs = dofs[:, :7]

		#self.samples[traj] = dofs

		return dofs	

	def plotTraj(self):
		waypoints = self.sampleWaypoints(self.waypts)
		self.plotWaypoints(self.env, self.robot, waypoints)

	def plotWaypoints(self, env, robot, waypoints):
		bodies = []
		for waypoint in waypoints:
			dof = np.append(waypoint, np.array([1, 1, 1]))
			coord = transformToCartesian(dofToTransform(robot, dof))
			plotPoint(env, bodies, coord)

		return bodies

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
	g = np.array(goal_pos)*(math.pi/180.0)

	s = np.array([-1, 2, 0, 2, 0, 4, 0])
	g = np.array([0,  2.9 ,  0.0 ,  2.1 ,  0. ,  4. ,  0.])

	print s
	print g

	T = 2.0
	trajplanner = Planner(s,g,T)
	trajplanner.plotTraj()
	t = 0.8
	theta = trajplanner.interpolate(t)
	

	trajplanner.execute_path_sim()

