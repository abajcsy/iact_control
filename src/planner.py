import numpy as np
import math
from sympy import symbols
from numpy import linspace
from sympy import lambdify
import matplotlib.pyplot as plt
import time

class PathPlanner(object):
	"""
	This class represents simple continous, time-parametrize path planning 
	class.

	Required parameters:
		start    - start cofiguration (at t = 0)
		goal	 - goal configuration (at t = T)
		T 		 - total time for trajectory execution
	"""

	def __init__(self, start, goal, T):
		self.s = start
		self.g = goal
		self.T = T

	def linear_path(self,t):
		"""
		Returns linear (in C-space) time-parametrized lambda function for each joint
		"""
		theta = (self.g-self.s)*(1/self.T)*t + self.s

		# if time after the final time, then just go to goal
		if t > self.T:
			theta = self.g

		return theta

	def update_T(self, newT):
		"""
		Updates T to rescale linear path.
		"""
		self.T = newT

if __name__ == '__main__':
	T = 20.0
	s = np.array([180]*7).reshape((7,1))* (math.pi/180.0)
	g = np.array([103.366,197.13,180.070,43.4309,265.11,257.271,287.9276]).reshape((7,1))* (math.pi/180.0)

	planner = PathPlanner(s,g,T)
	print planner.linear_path(10.4)

