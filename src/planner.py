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

	
	def time_trajectory(self, t, waypts, deltaT):
		"""
		Times a trajectory made from multiple waypts with parabolic blends
		- Method: 	Linear functions with parabolic blends (LFPB)
		- Ref: 		https://smartech.gatech.edu/bitstream/handle/1853/41948/ParabolicBlends.pdf
		"""
		# TODO THIS IS WRONG IN THE LONG RUN, JUST EXPERIMENTAL
		t_b = [1.0]*len(waypts)
		t_f = t_b[0]/2.0 + (deltaT[0]+deltaT[1]) + t_b[1]/2.0 
		v = [0, (waypts[1]-waypts[0])/deltaT[0], (waypts[2]-waypts[1])/deltaT[1], 0]
		a = [0, (v[1]-v[0])/t_b[0], (v[2]-v[1])/t_b[1], 0]

		print "Time to execute traj, t_f: " + str(t_f) 

		# compute time, T_i for each of the waypoints i in {1,...n}
		T = [None]*len(waypts)
		for i in range(len(waypts)):
			T_sum = 0.0
			for j in range(i):
				T_sum += deltaT[j]
			T[i] = t_b[0]/2.0 + T_sum
		print "T:" + str(T) 

		theta = np.array([0.0]*7).reshape((7,1))

		blend_idx = -1
		# check if in blend phase (and if yes, which one)
		print "t: " + str(t)
		for i in range(len(waypts)):
			if t >= (T[i]-t_b[i]/2.0) and t < (T[i]+t_b[i]/2.0):
				blend_idx = i

		lin_idx = -1
		# check if in linear phase (and if yes, which one)
		for i in range(len(waypts)-1):
			if t >= (T[i]+t_b[i]/2.0) and t < (T[i+1]-t_b[i+1]/2.0):
				lin_idx = i

		print "blend_idx: " + str(blend_idx) + ", lin_idx: " + str(lin_idx)

		# in blend phase
		if blend_idx != -1 and lin_idx == -1:
			print "blend phase"
			i = blend_idx
			theta = waypts[i] + v[i]*(t-T[i])+0.5*a[i]*(t-T[i]+(t_b[i]/2.0))**2
		elif blend_idx == -1 and lin_idx != -1:
		# in linear phase
			print "linear phase"
			i = lin_idx
			theta = waypts[i] + v[i+1]*(t-T[i])
		elif blend_idx == -1 and lin_idx == -1:
			print "past goal time"
			theta = waypts[-1] # just set last waypoint as goal
		
		print "theta: " + str(theta)
		return (self.T, theta)			

if __name__ == '__main__':
	T_total = 5.0
	alpha = 1
	s = np.array([180]*7).reshape((7,1))* (math.pi/180.0)
	g = np.array([103.366,197.13,180.070,43.4309,265.11,257.271,287.9276]).reshape((7,1))*(math.pi/180.0)

	waypt1 = np.array([136.886, 200.805, 64.022, 116.637, 138.328, 122.469, 179.861]).reshape((7,1))*(math.pi/180.0)
	waypt2 = np.array([271.091, 225.708, 20.548, 158.572, 160.879, 183.520, 186.644]).reshape((7,1))*(math.pi/180.0)
	waypt3 = np.array([338.680, 172.142, 25.755, 96.798, 180.497, 137.340, 186.655]).reshape((7,1))*(math.pi/180.0)

	waypts = [waypt1, waypt2, waypt3]
	deltaT = [2.0, 2.0]
	print "waypt1:"+str(waypt1)
	print "waypt2:"+str(waypt2)
	print "waypt3:"+str(waypt3)
	planner = PathPlanner(s,g,T_total,alpha)
	(T_total, theta) = planner.LFPB(2.0)
	(T_total, theta) = planner.time_trajectory(4.5, waypts, deltaT)
	print theta

