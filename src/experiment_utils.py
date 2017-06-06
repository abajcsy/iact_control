import numpy as np
from numpy import linalg
from numpy import linspace
import matplotlib.pyplot as plt
import time
import scipy
import math

#import or_trajopt
#import openravepy
#from openravepy import *

import logging
import copy

import csv

# measure total trajectory time
# measure torque_H over time
# measure interaction time (total)
# measure smoothness

class ExperimentUtils(object):
	
	def __init__(self):
		# stores waypoints of originally planned trajectory
		self.original_traj = None
		# stores the list of positions as the robot is moving 
		self.tracked_traj = np.array([0.0]*7)
		# stores start and end time of the interaction
		self.startT = 0.0
		self.endT = 0.0
		# stores running list of forces applied by human
		self.tauH = np.array([0.0]*7)

	def update_traj(self, curr_pos):
		#uses current position read from the robot to update the trajectory 
		self.tracked_traj = np.vstack([self.tracked_traj, curr_pos.reshape(7)])
		
	def update_tauH(self,tau_h):
		self.tauH = np.vstack([self.tauH, tau_h.reshape(7)])
	
	def set_startT(self,start_t):
		self.startT = start_t

	def set_endT(self,end_t):
		self.endT = end_t
		
	def plot_tauH(self):
		c = ['b','g','r','c','m','y','#FF8C00']
		tauT = [i for i in range(len(self.tauH))]

		fig, ax = plt.subplots(nrows=7, ncols=1, sharex=True, sharey=True)
		
		fig.text(0.5, 0.92, 'Human Interaction Forces During Experiment', ha='center' )
		fig.text(0.5, 0.04, 'time (t)', ha='center')
		fig.text(0.04, 0.5, 'human force (Nm)', va='center', rotation='vertical')

		# plot joint_torques
		for i in range(7):		
			ax = plt.subplot(7, 1, i+1)
			# plot the joint torque over time
			print str(i) + "joint tau: " + str(self.tauH[:,i]) 
			base_line,  = plt.plot(tauT, self.tauH[:,i], '-', linewidth=3.0, color=c[i])
			ax.set_ylim([-30,30])
			ax.set_xlim([0, len(self.tauH)])
			ax.set_yticks(scipy.arange(-30,31,15))
			plt.ylabel("joint " + str(i+1))
			ax.get_xaxis().set_visible(False)
			if i == 6:
				ax.get_xaxis().set_visible(True)	

		#plt.grid()
		
		plt.show()
		return

if __name__ == '__main__':

	experi = ExperimentUtils()
	tau_h = np.array([1,2,3,4,5,6,7])
	experi.update_tauH(tau_h)
	experi.plot_tauH()
	print experi.tauH
