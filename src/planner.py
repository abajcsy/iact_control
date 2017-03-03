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
		Returns time-dependant configuratrion for straight line trajectory. 
		- Method: 	1st order time-parametrized function
		"""
		#self.s = curr_pos
		#self.T = (linalg.norm(self.s-self.g)**2)*self.alpha

		theta = (self.g-self.s)*(1/self.T)*t + self.s

		# if time after the final time, then just go to goal
		if t > self.T:
			theta = self.g

		return (self.T, theta)

	def third_order_linear(self, t, curr_pos):
		"""
		Returns time-dependant configuratrion for straight line trajectory. 
		- Method: 	3rd order time-parametrized function
		"""
		c0 = self.s
		c1 = 0
		c2 = 3*(self.g-self.s)/(self.T**2)
		c3 = -2*(self.g-self.s)/(self.T**3)

		theta = c0 + c1*t + c2*t**2 +c3*t**3

		# if time after the final time, then just go to goal
		if t > self.T:
			theta = self.g

		return (self.T, theta)

	def LFPB(self, t):
		"""
		Returns time-dependant configuratrion for straight line trajectory. 
		- Method: 	Linear function with parabolic blend (LFPB)
		"""
		v = 2*(self.g-self.s)/self.T 				# max velocity	
		t_b = (self.s-self.g+v*self.T)/v 			# blending time

		a = self.s
		b = 0.0
		c = v/(2*t_b)
		d = (self.s+self.g-v*self.T)/2.0
		e = self.g
		f = 0.0
		g = (v*self.T+self.s-self.g+2.0*t_b*(-v))/(2.0*t_b**2)

		theta = np.array([0.0]*7).reshape((7,1))
		
		for i in range(7):
			if t >= 0.0 and t < t_b[i][0]:
				theta[i][0] = a[i][0]+t*b+(t**2)*c[i][0]
			elif t >= t_b[i][0] and t < self.T-t_b[i][0]:
				theta[i][0] = d[i][0] + v[i][0]*t
			elif t >= self.T-t_b[i][0] and t <= self.T:
				theta[i][0] = e[i][0]+(t-self.T)*f+((t-self.T)**2)*g[i][0]
			else: # if t > self.T
				theta[i][0] = self.g[i][0]
		
		return (self.T, theta)

	def update_T(self, newT):
		"""
		Updates T to rescale linear path.
		"""
		self.T = newT

if __name__ == '__main__':
	T = 5.0
	alpha = 1
	s = np.array([180]*7).reshape((7,1))* (math.pi/180.0)
	g = np.array([103.366,197.13,180.070,43.4309,265.11,257.271,287.9276]).reshape((7,1))* (math.pi/180.0)

	planner = PathPlanner(s,g,T,alpha)
	(T, theta) = planner.LFPB(2.0)
	print theta

