import numpy as np
from numpy import linalg
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
		alpha    - scaling factor on total trajectory time 
	"""

	def __init__(self, start, goal, T, alpha):
		self.s = start
		self.g = goal
		self.T = T
		self.alpha = alpha

	def linear_path(self,t,curr_pos):
		"""
		Returns linear (in C-space) time-parametrized lambda function for each joint
		"""
		#self.s = curr_pos
		#self.T = (linalg.norm(self.s-self.g)**2)*self.alpha

		theta = (self.g-self.s)*(1/self.T)*t + self.s

		# if time after the final time, then just go to goal
		if t > self.T:
			theta = self.g

		return (self.T, theta)

	def update_T(self, newT):
		"""
		Updates T to rescale linear path.
		"""
		self.T = newT

if __name__ == '__main__':
	T = 1.0
	s = np.array([180]*7).reshape((7,1))* (math.pi/180.0)
	g = np.array([103.366,197.13,180.070,43.4309,265.11,257.271,287.9276]).reshape((7,1))* (math.pi/180.0)

	planner = PathPlanner(s,g,T)
	print planner.linear_path(10.4)

