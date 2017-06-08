import numpy as np
from numpy import linalg
from numpy import linspace
import pandas as pd
from matplotlib import rc
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

		# stores final weights
		self.weights = [None]

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
		"""
		Records start time for experiment
		"""
		self.startT = start_t

	def set_endT(self,end_t):
		"""
		Records end time for experiment
		"""
		self.endT = end_t


	# ---- Plotting Functionality ---- #

	def plot_tauH(self, ID, task, method):
		"""
		Plots human-applied force from data file specified by:
			force<ID><task><method>.csv
		"""
		data = self.parse_data("force")

		values = data[ID][task][method]
		tauT = values[0]
		tauH = values[1:8]

		# fonts
		rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
		rc('text', usetex=True)

		# colors 
		c = ['b','g','r','c','m','y','#FF8C00']
	
		fig, ax = plt.subplots(nrows=7, ncols=1, sharex=True, sharey=True)
		
		fig.text(0.5, 0.92, 'Human Interaction Forces During Experiment', ha='center', fontsize=20)
		fig.text(0.5, 0.04, 'Time (s)', ha='center', fontsize=18)
		fig.text(0.04, 0.5, '$u_H$ (Nm)', va='center', rotation='vertical', fontsize=18)

		# plot joint_torques
		for i in range(7):		
			ax = plt.subplot(7, 1, i+1)
			# plot the joint torque over time
			base_line,  = plt.plot(tauT, tauH[i], '-', linewidth=3.0, color=c[i])
		
			ax.set_ylim([-30,30])
			ax.set_yticks(scipy.arange(-30,31,15))
			plt.ylabel("Joint " + str(i+1), fontsize=15)

			ax.set_xlim([0, np.amax(tauT)])
			# remove x-axis number labels
			ax.get_xaxis().set_visible(False)

			# remove the plot frame lines
			ax.spines["top"].set_visible(False)    
			ax.spines["right"].set_visible(False)      
			ax.spines["bottom"].set_visible(False)   

			if i == 6:
				# only for the last graph show the bottom axis
				ax.get_xaxis().set_visible(True)
				ax.spines["bottom"].set_visible(True)  	

			# Turn off tick labels
			ax.xaxis.set_ticks_position('none') 
			ax.yaxis.set_ticks_position('none') 

		plt.show()

	def plot_tauH_together(self, ID, task, method):
		"""
		Plots human-applied force from data file specified by:
			force<ID><task><method>.csv
		"""
		data = self.parse_data("force")

		values = data[ID][task][method]
		tauT = values[0]
		tauH = values[1:8]

		# fonts
		rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
		rc('text', usetex=True)

		# colors 
		c = ['b','g','r','c','m','y','#FF8C00']
	
		fig, ax = plt.subplots()
		
		fig.text(0.5, 0.92, 'Human Interaction Forces During Experiment', ha='center', fontsize=20)
		fig.text(0.5, 0.04, 'Time (s)', ha='center', fontsize=18)
		fig.text(0.04, 0.5, '$u_H$ (Nm)', va='center', rotation='vertical', fontsize=18)

		baselines = []
		# plot joint_torques
		for i in range(7):		
			# plot the joint torque over time
			base_line,  = plt.plot(tauT, tauH[i], '-', linewidth=3.0, color=c[i])
			baselines.append(base_line)

			ax.set_ylim([-30,30])
			ax.set_yticks(scipy.arange(-30,31,15))
			plt.ylabel("Joint " + str(i+1), fontsize=15)

			ax.set_xlim([0, np.amax(tauT)])
			# remove x-axis number labels
			ax.get_xaxis().set_visible(False)

			# remove the plot frame lines
			ax.spines["top"].set_visible(False)    
			ax.spines["right"].set_visible(False)      
			ax.spines["bottom"].set_visible(False)   

			if i == 6:
				# only for the last graph show the bottom axis
				ax.get_xaxis().set_visible(True)
				ax.spines["bottom"].set_visible(True)  	

			# Turn off tick labels
			ax.xaxis.set_ticks_position('none') 
			ax.yaxis.set_ticks_position('none') 

		l = []
		tex = []
		for i in range(7):
			l.append(baselines[i])
			tex.append("J"+str(i))
		leg = ax.legend(l, tex, fontsize=15)
		leg.get_frame().set_linewidth(0.0)

		plt.show()

	def plot_avgEffort(self, saveFig=False):
		"""
		Takes all participant data files and produces bar chart
		comparing average force exerted by each participant for trial
		with Method A or Method B.
		"""

		data = self.parse_data("force")

		N = len(data.keys()) # number participants

		A_means = [0.0]*N
		A_std = [0.0]*N

		B_means = [0.0]*N
		B_std = [0.0]*N

		for ID in data.keys():
			for task in range(len(data[ID])):
				Avalues = data[ID][task]['A']
				Bvalues = data[ID][task]['B']
				# only compute metrics over data, not timestamps
				Adata = Avalues[1:8]
				Bdata = Bvalues[1:8]

				A_means[ID-1] = np.mean(Adata)
				A_std[ID-1] = np.std(Adata)

				B_means[ID-1] = np.mean(Bdata)
				B_std[ID-1] = np.std(Bdata)
		
		print A_means
		print A_std

		ind = np.arange(N)  # the x locations for the groups
		width = 0.45       # the width of the bars

		# colors
		redC = (214/255., 39/255., 40/255.)
		greenC = (44/255., 160/255., 44/255.)
		blueC = (31/255., 119/255., 180/255.)
		orangeC = (255/255., 127/255., 14/255.)

		# fonts
		rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
		rc('text', usetex=True)

		fig, ax = plt.subplots()
		rectsA = ax.bar(ind, A_means, width, color=greenC, ecolor='k', edgecolor="none")
		rectsB = ax.bar(ind + width, B_means, width, color=orangeC, ecolor='k', edgecolor="none")

		# plots with stdev
		#rectsA = ax.bar(ind, A_means, width, color=greenC, yerr=A_std, ecolor='k', edgecolor="none")
		#rectsB = ax.bar(ind + width, B_means, width, color=orangeC, yerr=B_std, ecolor='k', edgecolor="none")

		def autolabel(rects):
			"""
			Attach a text label above each bar displaying its height
			"""
			for rect in rects:
				height = rect.get_height()
				ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,'%.2f' % 
						height,ha='center', va='bottom', fontsize=15)

		autolabel(rectsA)
		autolabel(rectsB)

		# add some text for labels, title and axes ticks
		ax.set_ylabel('Avg Force',fontsize=15)
		ax.set_xlabel('Participant Number',fontsize=15)
		ax.set_title('Average Human Effort for Experiment 1',fontsize=22)
		ax.set_xticks(ind + width)

		xlabels = ["P"+str(i+1) for i in range(N)]
		ax.set_xticklabels(xlabels,fontsize=15)

		# remove the plot frame lines
		ax.spines["top"].set_visible(False)    
		ax.spines["right"].set_visible(False)      
		ax.spines["bottom"].set_visible(False)    
		ax.spines["left"].set_visible(False)    

		plt.ylim(0, 6)    
		# Turn off tick labels
		ax.set_yticklabels([])

		# ensure that the axis ticks only show up on left of the plot.  
		ax.xaxis.set_ticks_position('none') 
		ax.yaxis.set_ticks_position('none') 
		#ax.get_yaxis().tick_left()   		

		leg = ax.legend((rectsA[0], rectsB[0]), ('Impedance', 'Learning'), fontsize=15)
		leg.get_frame().set_linewidth(0.0)
		plt.show()

		if saveFig:		
			plt.savefig("avgEffort.png", bbox_inches="tight")


	# ---- I/O Functionality ---- #

	def parse_data(self, dataType):
		"""
		Parses a set of CSV files from multiple participants and returns
		an aggregate array of the data for all participants across all trials

		Returns:
		ID 1 --> [Task1 --> [Method A, Method B], Task2 --> [Method A, Method B]]
		ID 2 --> [Task1 --> [Method A, Method B], Task2 --> [Method A, Method B]]
		...
		ID N --> [Task1 --> [Method A, Method B], Task2 --> [Method A, Method B]]
		---
		dataType - can be 'force' or 'traj'
		"""

		here = os.path.dirname(os.path.realpath(__file__))
		subdir = "/data/experimental/"
		datapath = here + subdir

		if dataType != "force" and dataType != "traj":
			print dataType + " is not a valid data type!"
			return None

		data = {}
		for filename in os.listdir(datapath):
			if dataType in filename:
				info = filename.split(dataType)[1]
				ID = int(info[0])
				task = int(info[1])
				methodType = info[2]

				# sanity check if participant ID and experi# exist already
				if ID not in data:
					data[ID] = {}
				if task not in data[ID]:
					data[ID][task] = {}
				with open(os.path.join(datapath, filename), 'r') as f:
					methodData = [None]*8
					i = 0
					for line in f:
						values = line.split(',')
						final_values = [float(v) for v in values[1:len(values)]]
						methodData[i] = final_values
						i += 1
					data[ID][task][methodType] = np.array(methodData)
				print "f: " + str(filename) + "|" + str(data)

		print data
		return data				
		

	def save_tauH(self, filename):
		"""
		Saves the human-applied force data to CSV file. 
		"""	
		# get the current script path
		here = os.path.dirname(os.path.realpath(__file__))
		subdir = "/data/experimental/"
		filepath = here + subdir + filename 

		print self.tauH
		print self.tauH.T

		with open(filepath, 'w') as out:
			# write total interaction time for each joint 
			#total_itime = 0.0
			#prevT = -1.0
			#for t in range(len(self.tauH)):
			#	if np.sum(self.tauH[t][1:8]) != 0:
			#		if prevT != -1.0:
			#			total_itime += (self.tauH[t][0] - prevT) 
			#		prevT = self.tauH[t][0]
			#out.write('total_iactT: %f\n' % total_itime)
			
			# write total interaction force for each joint
			#total_iforce = np.sum(self.tauH[:,1:8], axis=0)
			#out.write('total_iactF: %s\n' % total_iforce)

			# write headers for data
			#out.write('time,tau_j1,tau_j2,tau_j3,tau_j4,tau_j5,tau_j6,tau_j7\n')
			for i in range(len(self.tauH.T)):
				if i == 0:
					out.write('time')
				else:
					out.write('tau_j'+str(i))
				#out.write('%f' % self.tauH[t][0])
				for j in range(len(self.tauH.T[i])):
					out.write(',%f' % self.tauH.T[i][j])
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
			out.write('total_trajT: %f\n' % (self.endT-self.startT))
			out.write('time,j1,j2,j3,j4,j5,j6,j7\n')
			for t in range(len(self.tracked_traj)):
				out.write('%f' % self.tracked_traj[t][0])
				for j in range(1,len(self.tracked_traj[t])):
					out.write(',%f' % self.tracked_traj[t][j])
				out.write('\n')
		out.close()

if __name__ == '__main__':

	experi = ExperimentUtils()

	#tau_h = np.array([0]*7)
	#experi.update_tauH(5.4, tau_h)
	#tau_h = np.array([0,1,1,6,7,8,9])
	#experi.update_tauH(10.2, tau_h)
	#tau_h = np.array([0,0,1,0,0,0,0])
	#experi.update_tauH(11.2, tau_h)
	#print experi.tauH
	#experi.save_tauH("force20B.csv")
	experi.parse_data("force")
	experi.plot_avgEffort()
	experi.plot_tauH(3, 0, 'B')
	#experi.plot_tauH_together(3, 0, 'B')
