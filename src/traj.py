import numpy as np
from numpy import linalg
from numpy import linspace
import time
import math

from sympy import symbols
from sympy import lambdify

import trajoptpy
import or_trajopt

import interactpy
from interactpy import *

import logging
import pid
import copy

class Trajectory(object):
	"""
	This class represents a trajectory from start to goal.

	Required parameters:
		waypts   - untimed raw waypoints from Planner
		T 		 - total time for trajectory execution
		features - list of trajectory features
		weights  - list of weights for trajectory features
	"""

	def __init__(self):

		self.totalT = None
		self.num_waypts = None
		self.curr_waypt_idx = None
		self.waypts = None
		self.wayptsT = None
		self.raw_waypts = None

	def featurize(self, traj, totalT):
		"""
		Returns an array of feature values along traj
		using specified features 
		"""
		# TODO
		return 

	def update_weights(self, oldTraj, newTraj):
		"""
		Returns updated weights for new trajectory 
		given old trajectory
		"""

		
		#TODO
		return

if __name__ == '__main__':
	
	candlestick = np.array([180.0]*7)	
	home_pos = np.array([103.366,197.13,180.070,43.4309,265.11,257.271,287.9276])

	s = np.array(home_pos)*(math.pi/180.0)
	g = np.array(candlestick)*(math.pi/180.0)

	# test deformations
	u_h = np.array([0, 20, 0, 20, 20, 0, 0]).reshape((7,1))
	
