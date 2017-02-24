import numpy as np
import math
from sympy import symbols
from numpy import linspace
from sympy import lambdify
import matplotlib.pyplot as plt
import time

class PathPlanner(object):

	def __init__(self, start, goal, T):
		self.s = start
		self.g = goal
		self.T = T

	def linear_path(self,t):
		"""
		Returns linear (in C-space) time-parametrized lambda function for each joint
		"""
		print "s:"+str(self.s)+", g:" + str(self.g) + ", T:" + str(self.T)
		print "g-s:"+str(self.g-self.s)
		theta = (self.g-self.s)*(1/self.T)*t + self.s
		print "theta:" + str(theta)

		return theta

if __name__ == '__main__':
	T = 20.0
	s = np.array([180]*7).reshape((7,1))* (math.pi/180.0)
	g = np.array([103.366,197.13,180.070,43.4309,265.11,257.271,287.9276]).reshape((7,1))* (math.pi/180.0)

	planner = PathPlanner(s,g,T)
	print planner.linear_path(10.4)

	"""
	for i in range(7):
		l = "j"+str(i)

		x_vals = np.linspace(0,20,500)
		y_vals = planner.linear_path(x_vals)

		plt.plot(x_vals, y_vals, label=l)

	plt.plot(x_vals, y_vals, label=l)

	plt.xlabel('time (s)')
	plt.ylabel('theta')
	plt.legend()
	plt.show()
	"""

