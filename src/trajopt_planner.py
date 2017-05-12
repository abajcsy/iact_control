import numpy as np
from numpy import linalg
import math
from sympy import symbols
from numpy import linspace
from sympy import lambdify
import matplotlib.pyplot as plt
import time

import openravepy
import trajoptpy
import or_trajopt

import interactpy
from openravepy import *
from interactpy import initialize
from interactpy import demo
from interactpy import interact 

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

		#time.sleep(20)

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

		self.robot.SetDOFValues(self.s)
		orig_ee = self.robot.arm.hand.GetTransform()
		
		self.waypts = self.robot.arm.PlanToConfiguration(self.g)
		self.num_waypts = self.waypts.GetNumWaypoints()
		for i in range(self.num_waypts):
			print self.waypts.GetWaypoint(i)

		# compute time, T_i for each of the waypoints i in {1,...n}
		self.wayptsT = [None]*self.num_waypts
		T_sum = 0.0
		for i in range(self.num_waypts):
			self.wayptsT[i] = T_sum
			T_sum += self.totalT/self.num_waypts

	def interpolate(self, t):
		"""
		Gets the next desired position along trajectory
		by interpolating between waypoints given the current t.
		"""
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
					return theta
				# if exactly at a waypoint, return that waypoint
				elif t == self.wayptsT[i]:
					return self.waypts.GetWaypoint(i)			
		else:
			print "ONLY ONE WAYPT, CAN'T INTERPOLATE."
			return self.GetWaypoint(0)

	def LFPB(self, t):
		"""
		Returns time-dependant configuratrion for straight line trajectory. 
		- Method: 	Linear function with parabolic blend (LFPB)
		"""
		v = 2*(self.g-self.s)/self.t_f				# max velocity	
		t_b = (self.s-self.g+v*self.t_f)/v 			# blending time

		a = self.s
		b = 0.0
		c = v/(2*t_b)
		d = (self.s+self.g-v*self.t_f)/2.0
		e = self.g
		f = 0.0
		g = (v*self.t_f+self.s-self.g+2.0*t_b*(-v))/(2.0*t_b**2)

		theta = np.array([0.0]*7).reshape((7,1))
		
		for i in range(7):
			if t >= 0.0 and t < t_b[i][0]:
				theta[i][0] = a[i][0]+t*b+(t**2)*c[i][0]
			elif t >= t_b[i][0] and t < self.t_f-t_b[i][0]:
				theta[i][0] = d[i][0] + v[i][0]*t
			elif t >= self.t_f-t_b[i][0] and t <= self.t_f:
				theta[i][0] = e[i][0]+(t-self.t_f)*f+((t-self.t_f)**2)*g[i][0]
			else: # if t > self.T
				theta[i][0] = self.g[i][0]
		
		return theta

if __name__ == '__main__':
	
	home = np.array([103.366,197.13,180.070,43.4309,265.11,257.271,287.9276])
	goal = np.array([338.680, 172.142, 25.755, 96.798, 180.497, 137.340, 186.655])
	candlestick = np.array([180.0]*7)	
	
	waypt1 = np.array([136.886, 200.805, 64.022, 116.637, 138.328, 122.469, 179.861])
	waypt2 = np.array([271.091, 225.708, 20.548, 158.572, 160.879, 183.520, 186.644])
	waypt3 = np.array([338.680, 172.142, 25.755, 96.798, 180.497, 137.340, 186.655])

	home_pos = home
	goal_pos = goal

	padding = np.array([0,0,0])
	home_pos = np.append(home_pos, padding, 1)

	s = np.array(home_pos)*(math.pi/180.0)
	g = np.array(goal_pos)*(math.pi/180.0)

	print s
	print g

	T = 2.0
	trajplanner = Planner(s,g,T)
	
	P = np.array([[50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
			 [0.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0],
			 [0.0, 0.0, 50.0, 0.0, 0.0, 0.0, 0.0],
			 [0.0, 0.0, 0.0, 50.5, 0.0, 0.0, 0.0],
			 [0.0, 0.0, 0.0, 0.0, 50.0, 0.0, 0.0],
			 [0.0, 0.0, 0.0, 0.0, 0.0, 50.0, 0.0],
			 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 50.0]])
	I = 0.0*np.eye(7)
	D = np.array([[20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
			 [0.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0],
			 [0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0],
			 [0.0, 0.0, 0.0, 50.5, 0.0, 0.0, 0.0],
			 [0.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0],
			 [0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.0],
			 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0]])
	controller = pid.PID(P,I,D,0,0)

	t = 0.8
	theta = trajplanner.interpolate(t)
	print "theta: " + str(theta)
	p_error = (goal - theta).reshape((7,1))*(math.pi/180.0)
	print "perror: " + str(p_error)
	tau = controller.update_PID(p_error)
	print "tau: " + str(tau)


	trajplanner.execute_path_sim()

