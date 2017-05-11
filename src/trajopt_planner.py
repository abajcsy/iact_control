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

import logging

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
		self.s = start
		self.g = goal
		self.totalT = T

		# initialize openrave and compute waypts
		#model_filename = 'jaco_original'
		env, robot = initialize()

		viewer = env.GetViewer()
		viewer.SetCamera([[ 0.94684722, -0.12076704,  0.29815376,  0.21004671],
		   [-0.3208323 , -0.42191214,  0.84797216, -1.40675116],
		   [ 0.0233876 , -0.89855744, -0.43823231,  1.04685986],
		   [ 0.        ,  0.        ,  0.        ,  1.        ]])

		
		# set the starting DOFs
		robot.SetDOFValues(self.s)
		orig_ee = robot.arm.hand.GetTransform()
		print "g: " + str(self.g)
		
		waypts = robot.arm.PlanToConfiguration(self.g)
		num_waypts = waypts.GetNumWaypoints()
		for i in range(num_waypts):
			print waypts.GetWaypoint(i)

		# compute time, T_i for each of the waypoints i in {1,...n}
		self.wayptsT = [None]*num_waypts
		T_sum = 0.0
		for i in range(num_waypts):
			self.wayptsT[i] = T_sum
			T_sum += self.totalT/num_waypts

		print "Waypoint times T:" + str(self.wayptsT) 

		robot.ExecutePath(waypts)
		#time.sleep(20)

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
		
		return (self.t_f, theta)

if __name__ == '__main__':
	
	
	home_pos = np.array([103.366,197.13,180.070,43.4309,265.11,257.271,287.9276])
	goal_pos = np.array([338.680, 172.142, 25.755, 96.798, 180.497, 137.340, 186.655])
	candlestick = np.array([180.0]*7)	

	waypt1 = np.array([136.886, 200.805, 64.022, 116.637, 138.328, 122.469, 179.861])
	waypt2 = np.array([271.091, 225.708, 20.548, 158.572, 160.879, 183.520, 186.644])
	waypt3 = np.array([338.680, 172.142, 25.755, 96.798, 180.497, 137.340, 186.655])

	home_pos = waypt1
	goal_pos = waypt3

	padding = np.array([0,0,0])
	home_pos = np.append(home_pos, padding, 1)

	#goal_pos = np.append(goal_pos, padding, 1)

	print home_pos
	print goal_pos
	s = np.array(home_pos)*(math.pi/180.0)
	g = np.array(goal_pos)*(math.pi/180.0)
	print s
	print g
	T = 2.0
	trajplanner = Planner(s,g,T)

