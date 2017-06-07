import numpy as np
from numpy import linalg
from numpy import linspace
import matplotlib.pyplot as plt
import time
import scipy
import math

import logging
import copy

import csv
import os

# measure total trajectory time
# measure torque_H over time
# measure interaction time (total)
# measure smoothness

class ExperimentUtils(object):
	
	def __init__(self):
		# stores waypoints of originally planned trajectory
		self.original_traj = None
		# stores the list of positions as the robot is moving 
		# in the form [timestamp, j1, j2, ... , j7]
		self.tracked_traj = np.array([0.0]*8)

		# stores start and end time of the interaction
		self.startT = 0.0
		self.endT = 0.0

		# stores running list of forces applied by human
		# in the form [timestamp, j1, j2, ... , j7]
		self.tauH = np.array([0.0]*8).reshape((1,8))

	def update_traj(self, timestamp, curr_pos):
		"""
		Uses current position read from the robot to update the trajectory
		Saves timestamp when this position was read
		""" 
		currTraj = np.append([timestamp], curr_pos.reshape(7))
		self.tracked_traj = np.vstack([self.tracked_traj, currTraj])
		
	def update_tauH(self, timestamp, tau_h):
		"""
		Uses current joint torque reading from the robot during interaction
		Saves timestamp when this torque was read
		""" 
		currTau = np.append([timestamp], tau_h.reshape(7))
		self.tauH = np.vstack([self.tauH, currTau])
	
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
		
		plt.show()
		return

	def plot_participantVsForce(self):
		"""
		NOTE: INCOMPLETE FUNCTION
		Takes all participant data files and produces bar chart
		comparing average force exerted by each participant for trial
		with Method A or Method B.
		"""
		# TODO read multiple data files here to aggregate data
		N = 5
		men_means = (20, 35, 30, 35, 27)
		men_std = (2, 3, 4, 1, 2)

		ind = np.arange(N)  # the x locations for the groups
		width = 0.45       # the width of the bars

		fig, ax = plt.subplots()
		rects1 = ax.bar(ind, men_means, width, color='g', yerr=men_std, ecolor='k', edgecolor="none")

		women_means = (25, 32, 34, 20, 25)
		women_std = (3, 5, 2, 3, 3)
		rects2 = ax.bar(ind + width, women_means, width, color='b', yerr=women_std, ecolor='k', edgecolor="none")

		# add some text for labels, title and axes ticks
		ax.set_ylabel('Avg Force')
		ax.set_xlabel('Participant Number')
		ax.set_title('Average Human Effort for Experiment 1')
		ax.set_xticks(ind + width)
		ax.set_xticklabels(('P1', 'P2', 'P3', 'P4', 'P5'))

		ax.legend((rects1[0], rects2[0]), ('Method A', 'Method B'))
		plt.show()

	def save_tauH(self, filename):
		"""
		Saves the human-applied force data to CSV file. 
		"""	
		# get the current script path
		here = os.path.dirname(os.path.realpath(__file__))
		subdir = "/data/experimental/"
		filepath = here + subdir + filename 

		with open(filepath, 'w') as out:
			out.write('total_trajT: %f\n' % (self.endT-self.startT))
			# write total interaction time for each joint 
			total_itime = 0.0
			prevT = -1.0
			for t in range(len(self.tauH)):
				if np.sum(self.tauH[t][1:8]) != 0:
					if prevT != -1.0:
						total_itime += (self.tauH[t][0] - prevT) 
					prevT = self.tauH[t][0]
			out.write('total_iactT: %f\n' % total_itime)
			# write total interaction force for each joint
			total_iforce = np.sum(self.tauH[:,1:8], axis=0)
			print total_iforce
			out.write('total_iactF: %s\n' % total_iforce)
			# write headers for trajectory data
			out.write('time,tau_j1,tau_j2,tau_j3,tau_j4,tau_j5,tau_j6,tau_j7\n')
			for t in range(len(self.tauH)):
				out.write('%f' % self.tauH[t][0])
				for j in range(1,len(self.tauH[t])):
					out.write(',%f' % self.tauH[t][j])
				out.write('\n')
		
		out.close()

	def save_trackedTraj(self, filename):
		"""
		Saves the measured positions of the trajectory to CSV file. 
		"""	

		# get the current script path
		here = os.path.dirname(os.path.realpath(__file__))
		subdir = "/data/experimental/"
		filepath = here + subdir + filename 

		with open(filepath, 'w') as out:
			out.write('time,j1,j2,j3,j4,j5,j6,j7\n')
			for t in range(len(self.tracked_traj)):
				out.write('%f' % self.tracked_traj[t][0])
				for j in range(1,len(self.tracked_traj[t])):
					out.write(',%f' % self.tracked_traj[t][j])
				out.write('\n')
		out.close()

if __name__ == '__main__':

	experi = ExperimentUtils()

	tau_h = np.array([0]*7)
	experi.update_tauH(5.4, tau_h)
	tau_h = np.array([1,2,3,4,5,6,7])
	experi.update_tauH(10.2, tau_h)
	tau_h = np.array([1,2,3,4,5,6,7])
	experi.update_tauH(11.2, tau_h)
	#experi.plot_tauH()
	print experi.tauH
	#experi.save_tauH('tauH_data.csv')
	experi.plot_participantVsForce()
