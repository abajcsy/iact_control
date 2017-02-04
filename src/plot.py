#! /usr/bin/env python
import rospy
import math

import numpy as np
import matplotlib.pyplot as plt

num_joints = 7

class Plotter(object):	
	"""
	This class implements basic plotting functionality for PID control. 
	
	
	Required parameters:
		p_gain, i_gain, d_gain    - gain terms for the PID controller
	"""
	def __init__(self,p,i,d):
		self.t = 0
		self.dt = 1

		self.p_gain = p
		self.i_gain = i
		self.d_gain = d

		self.times = np.zeros(1)
		self.p_error = np.zeros((7,1))
		self.i_error = np.zeros((7,1))
		self.d_error = np.zeros((7,1))


	def update_PID_plot(self, p_e, i_e, d_e):
		"""
		Updates the P,I,D errors based on most recent movement.
		"""
		self.p_error = np.column_stack((self.p_error,p_e))
		self.i_error = np.column_stack((self.i_error,i_e))
		self.d_error = np.column_stack((self.d_error,d_e))
		self.times = np.column_stack((self.times,np.array(self.t)))
		self.t += self.dt

	def plot_PID(self):
		"""
		Plots the P,I,D errors over time.
		"""
		# plot p_error
		plt.subplot(3, 1, 1)
		for i in range(num_joints):	
			l = "j"+str(i)
			plt.plot(self.times[0], self.p_error[i], '-', linewidth=3.0, label=l)
		plt.xlabel("time (s)")
		plt.ylabel("p error (rad)")
		plt.title("P,I,D error with K_p: " + str(self.p_gain) + ", K_i:" + str(self.i_gain) + ", K_d:" + str(self.d_gain))
		plt.legend()
		plt.grid()

		# plot i_error
		plt.subplot(3, 1, 2)
		for i in range(num_joints):	
			l = "j"+str(i)
			plt.plot(self.times[0], self.i_error[i], '-', linewidth=3.0, label=l)
		plt.xlabel("time (s)")
		plt.ylabel("i error")
		plt.legend()
		plt.grid()

		# plot d_error
		plt.subplot(3, 1, 3)
		for i in range(num_joints):	
			l = "j"+str(i)
			plt.plot(self.times[0], self.d_error[i], '-', linewidth=3.0, label=l)
		plt.xlabel("time (s)")
		plt.ylabel("d error")
		plt.legend()
		plt.grid()

		plt.show()
	

