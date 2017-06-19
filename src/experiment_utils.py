import numpy as np
from numpy import linalg
from numpy import linspace
from matplotlib import rc
import matplotlib.pyplot as plt
import time
import scipy
import math

import logging
import copy

import csv
import os

import openrave_utils
from openrave_utils import *


class ExperimentUtils(object):
	
	def __init__(self):
		# stores waypoints of originally planned trajectory
		self.original_traj = None
		# stores trajectory as it is deformed
		self.deformed_traj = None
		# stores the list of positions as the robot is moving 
		# in the form [timestamp, j1, j2, ... , j7]
		self.tracked_traj = None

		# stores start and end time of the interaction
		self.startT = 0.0
		self.endT = 0.0

		# stores weights over time 
		# always in the form [timestamp, weight]
		self.weights = None

		# stores running list of forces applied by human
		# in the form [timestamp, j1, j2, ... , j7]
		self.tauH = None 

	def update_original_traj(self, waypts):
		"""
		Updates the original trajectory
		"""
		self.original_traj = waypts

	def update_deformed_traj(self, waypts):
		"""
		Updates the deformed trajectory
		"""
		self.deformed_traj = waypts

	def update_tracked_traj(self, timestamp, curr_pos):
		"""
		Uses current position read from the robot to update the trajectory
		Saves timestamp when this position was read
		""" 
		currTraj = np.append([timestamp], curr_pos.reshape(7))
		if self.tracked_traj is None:
			self.tracked_traj = np.array([currTraj])
		else:	
			self.tracked_traj = np.vstack([self.tracked_traj, currTraj])
		
	def update_tauH(self, timestamp, tau_h):
		"""
		Uses current joint torque reading from the robot during interaction
		Saves timestamp when this torque was read
		""" 
		currTau = np.append([timestamp], tau_h.reshape(7))
		if self.tauH is None:
			self.tauH = np.array([currTau])
		else:
			self.tauH = np.vstack([self.tauH, currTau])

	def update_weights(self, timestamp, new_weight):
		"""
		Updates list of timestamped weights
		"""
		new_w = np.array([timestamp, new_weight])
		if self.weights is None:
			self.weights = np.array([new_w])
		else:
			self.weights = np.vstack([self.weights, new_w])
	
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

	# ---- Computational Functionality ---- #

	#def total_interaction_T(self, ID, task, method):
	#	"""
	#	Gets total interaction time for participant during task with method.
	#	"""
	#	data = self.parse_data("weights")
	#	values = data[ID][task][method]
	# 	TODO IMPLEMENT THIS!

		
	#def total_traj_T(self, ID, task, method):
	#	"""
	#	Gets total trajectory time for participant during task with method.
	#	"""
	#	#TODO IMPLEMENT THIS!

	#def total_interaction_F(self, ID, task, method): 
	#	"""
	#	Gets total force applied by participant during task with method.
	#	"""
	#	data = self.parse_data("weights")
	#	values = data[ID][task][method]
	#	TODO IMPLEMENT THIS!

	# ---- Plotting Functionality ---- #

	def plot_weights(self, ID, task, method,saveFig=False):
		data = self.parse_data("weights")
		values = data[ID][task][method]

		wT = values[0]
		wV = values[1]

		# fonts
		rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
		rc('text', usetex=True)

		# colors 
		c = ['b','g','r','c','m','y','#FF8C00']
	
		fig, ax = plt.subplots()
		
		fig.text(0.5, 0.92, 'Weight Update for Task ' + str(task), ha='center', fontsize=20)
		fig.text(0.5, 0.04, 'Time (s)', ha='center', fontsize=18)
		fig.text(0.04, 0.5, 'theta', va='center', rotation='vertical', fontsize=18)

		# plot the weights over time
		base_line,  = plt.plot(wT, wV, '-', linewidth=3.0, color=c[0])
	
		ax.set_ylim([0,1.1])
		ax.set_yticks(scipy.arange(0,1.1,0.5))
	
		# remove x-axis number labels
		#ax.get_xaxis().set_visible(False)

		# remove the plot frame lines
		ax.spines["top"].set_visible(False)    
		ax.spines["right"].set_visible(False)      
		#ax.spines["bottom"].set_visible(False) 
		#ax.spines["left"].set_visible(False)   

		# Turn off tick labels
		ax.xaxis.set_ticks_position('none') 
		ax.yaxis.set_ticks_position('none') 

		plt.show()

		if saveFig:
			here = os.path.dirname(os.path.realpath(__file__))
			subdir = "/data/experimental/"
			datapath = here + subdir
			fig.savefig(datapath+"theta"+str(ID)+str(task)+method+".png", bbox_inches="tight")
			print "Saved weight figure."

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

	def plot_avgEffort(self, saveFig=False, task=None):
		"""
		Takes all participant data files and produces bar chart
		comparing average force exerted by each participant for trial
		with Method A or Method B.
		----
		saveFig 	if True, saves final plot
		task		task number to plot avg force for (1,2,3)
					if task=None, then plots avg force over ALL 
					tasks (not including familiarization) 
		"""

		data = self.parse_data("force")

		N = len(data.keys()) # number participants

		A_means = [0.0]*N
		A_std = [0.0]*N

		B_means = [0.0]*N
		B_std = [0.0]*N

		if task is None:
			# average across ALL tasks
			for ID in data.keys():
				# dont include the familiarization task
				for t in range(1,len(data[ID])):
					Avalues = data[ID][t]['A']
					Bvalues = data[ID][t]['B']
					# only compute metrics over data, not timestamps
					Adata = Avalues[1:8]
					Bdata = Bvalues[1:8]

					A_means[ID] += np.mean(np.abs(Adata))
					B_means[ID] += np.mean(np.abs(Bdata))
				A_means[ID] /= 3.0
				B_means[ID] /= 3.0
		else:
			# average across SINGLE specified task
			for ID in data.keys():
				Avalues = data[ID][task]['A']
				Bvalues = data[ID][task]['B']
				
				Adata = Avalues[1:8]
				Bdata = Bvalues[1:8]

				A_means[ID] = np.mean(np.abs(Adata))
				B_means[ID] = np.mean(np.abs(Bdata))


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
		rectsA = ax.bar(ind, A_means, width, color=greenC, edgecolor="none")
		rectsB = ax.bar(ind + width, B_means, width, color=orangeC, edgecolor="none")

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
		ax.set_ylabel('Avg Effort (Nm)',fontsize=15)
		ax.set_xlabel('Participant Number',fontsize=15)
		print task
		if task is None:
			ax.set_title('Average Human Effort for All Experiments',fontsize=22)
		else:
			ax.set_title('Average Human Effort for Task '+str(task),fontsize=22)
		ax.set_xticks(ind + width)

		xlabels = ["P"+str(ID) for ID in data.keys()]
		ax.set_xticklabels(xlabels,fontsize=15)

		# remove the plot frame lines
		ax.spines["top"].set_visible(False)    
		ax.spines["right"].set_visible(False)      
		ax.spines["bottom"].set_visible(False)    
		ax.spines["left"].set_visible(False)    

		plt.ylim(0, 6)    
		# Turn off tick labels
		ax.set_yticklabels([])

		# set max y-limit to 2 Nm
		ax.set_ylim([0,2.0])

		# ensure that the axis ticks only show up on left of the plot.  
		ax.xaxis.set_ticks_position('none') 
		ax.yaxis.set_ticks_position('none') 
		#ax.get_yaxis().tick_left()   		

		leg = ax.legend((rectsA[0], rectsB[0]), ('Impedance', 'Learning'), fontsize=15)
		leg.get_frame().set_linewidth(0.0)
		plt.show()

		if saveFig:
			here = os.path.dirname(os.path.realpath(__file__))
			subdir = "/data/experimental/"
			datapath = here + subdir
			fig.savefig(datapath+"avgEffort.png", bbox_inches="tight")
			print "Saved effort figure."

	# ---- OpenRAVE Plotting ---- #

	def upsample(self, waypts_plan, start_time, final_time, step_time):
		"""
		Put waypoints along trajectory at step_time increments.
		---
		input desired time increment, update upsampled trajectory
		"""
		num_waypts = int(math.ceil((final_time - start_time)/step_time)) + 1
		waypts = np.zeros((num_waypts,7))
		waypts_time = [None]*num_waypts

		num_waypts_plan = 4 # len(waypts_plan)
		step_time_plan = (final_time - start_time)/(num_waypts_plan - 1)
		t = start_time
		for i in range(num_waypts):
			if t >= final_time:
				waypts_time[i] = final_time
				waypts[i,:] = waypts_plan[num_waypts_plan - 1]
			else:
				deltaT = t - start_time
				prev_idx = int(deltaT/step_time_plan)
				print prev_idx	
				print num_waypts			
				prev = waypts_plan[prev_idx]
				next = waypts_plan[prev_idx + 1]
				waypts_time[i] = t
				waypts[i,:] = prev+((t-prev_idx*step_time_plan)/step_time_plan)*(next-prev)
			t += step_time
		return waypts

	def plot_traj(self, dataType, filename):

		here = os.path.dirname(os.path.realpath(__file__))
		subdir = "/data/experimental/"
		datapath = here + subdir + dataType

		validTypes = ["deformed", "tracked", "original"]
		if dataType not in validTypes:
			print dataType + " is not a valid trajectory data type!"
			return None

		data = {}
		with open(os.path.join(datapath, filename), 'r') as f:
			# account for differently structured data between trajectories
			if dataType is not "tracked":
				lines = f.readlines()
			elif dataType is "tracked":
				 lines = f.readlines()[2:]
			jnum = 0
			for line in lines:
				values = line.split(',')
				final_values = [float(v) for v in values[1:len(values)]]
				data[jnum] = final_values
				jnum += 1

		# get waypoints from file data
		waypts = np.zeros((len(data[0]),7))
		for i in range(len(data[0])):
			jangles = []
			for j in range(7):
				jangles.append(data[j][i])
			waypts[i] = np.array(jangles)		

		print waypts		
		# load openrave
		model_filename = 'jaco_dynamics'
		env, robot = initialize(model_filename)

		# plot saved waypoints
		bodies = []
		plotTraj(env,robot,bodies,waypts)
		plotTable(env)
		plotTableMount(env,bodies)
		plotLaptop(env,bodies)
		plotCabinet(env)

		time.sleep(25)

	def plot_allTraj(self, ID, task, method, trial):
		"""
		Plots all three trajectories (tracked, deformed, original)
		for specified participant and trial and task with method
		"""
	
		if method is "B":
			file1 = "deformed" + str(ID) + str(task) + method + str(trial) + ".csv"
			# parse waypts from file
			waypts = self.parse_traj("deformed", file1)
			waypts = self.upsample(waypts, 0.0, 15.0, 0.5)

		# parse waypts from file
		file2 = "tracked" + str(ID) + str(task) + method + str(trial) + ".csv"
		waypts2 = self.parse_traj("tracked", file2)

		file3 = "original" + str(ID) + str(task) + method + str(trial) + ".csv"
		# parse waypts from file
		waypts3 = self.parse_traj("original", file3)
		waypts3 = self.upsample(waypts3, 0.0, 15.0, 0.5)

		# load openrave
		model_filename = 'jaco_dynamics'
		env, robot = initialize(model_filename)

		# plot saved waypoints
		bodies = []		
		if method is "B":
			plotTraj(env,robot,bodies,waypts)
		plotTraj(env,robot,bodies,waypts2,color=[1, 0, 0])
		plotTraj(env,robot,bodies,waypts3,color=[0, 0, 1])
		plotTable(env)
		plotTableMount(env,bodies)
		plotLaptop(env,bodies)
		#plotCabinet(env)

		time.sleep(25)

	# ---- Trajectory Analysis Functionality ----#

	def featurize_traj(self, dataType, filename):
		"""
		Takes trajectory file and featurizes it. 
		"""
		# parse waypts from file
		waypts = self.parse_traj(dataType, filename)

		# get task number
		values = filename.split(dataType)
		taskNum = int(values[1][1])

		print taskNum

		plan = Planner(taskNum)		
		features = plan.featurize(waypts)
 	
		print "features: " + str(features)
		return features	

	# ---- I/O Functionality ---- #

	def parse_traj(self, dataType, filename):
		here = os.path.dirname(os.path.realpath(__file__))
		subdir = "/data/experimental/"
		datapath = here + subdir + dataType

		validTypes = ["deformed", "tracked", "original"]
		if dataType not in validTypes:
			print dataType + " is not a valid trajectory data type!"
			return None

		data = {}
		with open(os.path.join(datapath, filename), 'r') as f:
			# account for differently structured data between trajectories
			if dataType is not "tracked":
				lines = f.readlines()
			elif dataType is "tracked":
				 lines = f.readlines()[2:]
			jnum = 0
			for line in lines:
				values = line.split(',')
				final_values = [float(v) for v in values[1:len(values)]]
				data[jnum] = final_values
				jnum += 1

		# get waypoints from file data
		waypts = np.zeros((len(data[0]),7))
		for i in range(len(data[0])):
			jangles = []
			for j in range(7):
				jangles.append(data[j][i])
			waypts[i] = np.array(jangles)		

		print "parsed trajectory: " + str(waypts)
		return waypts


	def parse_data(self, dataType):
		"""
		Parses a set of CSV files from multiple participants and returns
		an aggregate array of the data for all participants across all trials

		Returns:
		ID 1 --> [Task1 --> [Trial1 --> [Method A, Method B], Trial2 --> [Method A, Method B]], [Task2 --> ...]]
		ID 2 --> [Task1 --> [Trial1 --> [Method A, Method B], Trial2 --> [Method A, Method B]], [Task2 --> ...]]
		...
		ID N --> [Task1 --> [Trial1 --> [Method A, Method B], Trial2 --> [Method A, Method B]], [Task2 --> ...]]
		---
		dataType - can be 'force' or 'traj'
		"""

		here = os.path.dirname(os.path.realpath(__file__))
		subdir = "/data/experimental/"
		datapath = here + subdir + dataType

		validTypes = ["force", "weights", "deformed", "tracked", "original"]
		if dataType not in validTypes:
			print dataType + " is not a valid data type!"
			return None

		data = {}
		for filename in os.listdir(datapath):
			print filename
			info = filename.split(dataType)[1]
			ID = int(info[0])
			task = int(info[1])
			methodType = info[2]
			trial = int(info[3])

			# sanity check if participant ID and experi# exist already
			if ID not in data:
				data[ID] = {}
			if task not in data[ID]:
				data[ID][task] = {}
			if trial not in data[ID][task]:
				data[ID][task][trial] = {}
			with open(os.path.join(datapath, filename), 'r') as f:
				trajTypes = ["deformed", "original"]

				methodData = [None]*8
				if dataType == "weights": 
					methodData = [None]*2
				elif dataType in ["deformed", "original"]:
					methodData = [None]*7
				elif dataType is "tracked":
					methodData = [None]*8
				firstLine = True
				i = 0
				for line in f:
					# skip first line in tracked that has totalT
					if dataType is "tracked" and firstLine:
						firstLine = False
						continue
					values = line.split(',')
					final_values = [float(v) for v in values[1:len(values)]]
					methodData[i] = final_values
					i += 1
				#print methodData
				data[ID][task][trial][methodType] = np.array(methodData)

		return data				
		

	def save_tauH(self, filename):
		"""
		Saves the human-applied force data to CSV file. 
		"""	
		# get the current script path
		here = os.path.dirname(os.path.realpath(__file__))
		subdir = "/data/experimental/force/"
		filepath = here + subdir + filename + "1.csv"

		i = 2
		while os.path.exists(filepath):
			filepath = here+subdir+filename+str(i)+".csv"
			i+=1

	
		print "tau: " + str(self.tauH)
		#print "len(tau): " + str(len(self.tauH))

		with open(filepath, 'w') as out:
			if self.tauH is not None:
				for i in range(len(self.tauH.T)):
					if i == 0:
						out.write('time')
					else:
						out.write('tau_j'+str(i))
					for j in range(len(self.tauH.T[i])):
						out.write(',%f' % self.tauH.T[i][j])
					out.write('\n')
		out.close()

	def save_weights(self, filename):
		"""
		Saves the weights over time to CSV file. 
		"""	
		# get the current script path
		here = os.path.dirname(os.path.realpath(__file__))
		subdir = "/data/experimental/weights/"
		filepath = here + subdir + filename + "1.csv"

		i = 2
		while os.path.exists(filepath):
			filepath = here+subdir+filename+str(i)+".csv"
			i+=1

		print "weights: " + str(self.weights[:,0])
		#print "len(w): " + str(len(self.weights[:,0]))

		with open(filepath, 'w') as out:
			out.write('time')
			for i in range(len(self.weights[:,0])):
				out.write(',%f' % self.weights[i][0])		
			out.write('\n')					
			out.write('weights')
			for i in range(len(self.weights[:,1])):
				out.write(',%f' % self.weights[i][1])
			out.write('\n')
		out.close()

	def save_original_traj(self, filename):
		"""
		Saves the originally planned trajectory to CSV file.
		"""

		# get the current script path
		here = os.path.dirname(os.path.realpath(__file__))
		subdir = "/data/experimental/original/"
		filepath = here + subdir + filename + "1.csv"

		i = 2
		while os.path.exists(filepath):
			filepath = here+subdir+filename+str(i)+".csv"
			i+=1

		with open(filepath, 'w') as out:
			for j in range(7):
				out.write('j%d' % j)
				for pt in range(len(self.original_traj)):
					out.write(',%f' % self.original_traj[pt][j])
				out.write('\n')
		out.close()

	def save_deformed_traj(self, filename):
		"""
		Saves the deformed trajectory to CSV file.
		"""

		if self.deformed_traj is not None:
			# get the current script path
			here = os.path.dirname(os.path.realpath(__file__))
			subdir = "/data/experimental/deformed/"
			filepath = here + subdir + filename + "1.csv"

			i = 2
			while os.path.exists(filepath):
				filepath = here+subdir+filename+str(i)+".csv"
				i+=1

			with open(filepath, 'w') as out:
				for j in range(7):
					out.write('j%d' % j)
					for pt in range(len(self.deformed_traj)):
						out.write(',%f' % self.deformed_traj[pt][j])
					out.write('\n')
			out.close()
		else:
			print "No deformed trajectory to write."

	def save_tracked_traj(self, filename):
		"""
		Saves the measured positions of the trajectory to CSV file. 
		"""	

		# get the current script path
		here = os.path.dirname(os.path.realpath(__file__))
		subdir = "/data/experimental/tracked/"
		filepath = here + subdir + filename + "1.csv"

		i = 2
		while os.path.exists(filepath):
			filepath = here+subdir+filename+str(i)+".csv"
			i+=1

		print "sT: " + str(self.startT)
		print "eT: " + str(self.endT)
		with open(filepath, 'w') as out:
			out.write('total_trajT: %f\n' % (self.endT-self.startT))
			out.write('time')
			for t in range(len(self.tracked_traj[:,0])):
				out.write(',%f' % self.tracked_traj[t][0])		
			out.write('\n')					
			for j in range(1,8):
				out.write('j%d' % j)
				for waypt in self.tracked_traj[:,j]:
					out.write(',%f' % waypt)
				out.write('\n')
		out.close()

if __name__ == '__main__':

	experi = ExperimentUtils()
	#dataType = "tracked"	
	#filename = "tracked53B1.csv"
	#experi.plot_traj(dataType, filename)

	#experi.update_tauH(0.1, np.array([1,2,3,4,5,6,7]))
	#experi.update_tauH(0.2, np.array([1,2,3,4,5,6,7]))
	#experi.update_weights(0.1,1.0)
	#experi.update_weights(0.2,1.6)
	experi.plot_avgEffort(saveFig=False, task=1)
	#experi.plot_tauH(0, 2, 'B')
	#experi.plot_weights(0, 1, 'B',saveFig=False)
	#experi.plot_tauH_together(3, 0, 'B')
