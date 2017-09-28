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

import os
import pickle
import data_io

import openrave_utils
from openrave_utils import *

import experiment_utils
from experiment_utils import *

import trajopt_planner

# possible range of feature values for each feature
CUP_RANGE = 1.87608702
TABLE_RANGE = 0.6918574 
LAPTOP_RANGE = 1.00476554

pick_basic = [104.2, 151.6, 183.8, 101.8, 224.2, 216.9, 310.8]
pick_shelf = [210.8, 241.0, 209.2, 97.8, 316.8, 91.9, 322.8]
place_lower = [210.8, 101.6, 192.0, 114.7, 222.2, 246.1, 322.0]
place_higher = [210.5,118.5,192.5,105.4,229.15,245.47,316.4]

ONE_FEAT = 1					# experiment where human has to correct only one feature (out of three)
TWO_FEAT = 2					# experiment where human has to correc two features

ALL = "ALL" 							# updates all features
MAX = "MAX"								# updates only feature that changed the most

EXP_TASK = 2

# ------- Saves out cleaned and computed statistics ------# 

def	save_parsed_data(filename, csvData=True, pickleData=False):
	"""
	Restructures all data from participants into a single file for
	objective measures and subjective measures.
	-----
	Give the filename you want to save it as
	If you want to pickle or not
	"""
	obj_metrics = compute_obj_metrics()
	#subj_metrics = compute_subj_metrics()
	
	# write to file
	here = os.path.dirname(os.path.realpath(__file__))
	subdir = "/data/experimental/"


	if pickleData:
		filepath_obj = here + subdir + filename + "_obj.p"
		pickle.dump(obj_metrics, open( filepath_obj, "wb" ) )
		#filepath_subj = here + subdir + filename + "_subj.p"
		#pickle.dump(subj_metrics, open( filepath_subj, "wb" ) )

	if csvData:
		filepath_obj = here + subdir + filename + "_obj.csv"
		#filepath_subj = here + subdir + filename + "_subj.csv"

		# write objective metrics
		with open(filepath_obj, 'w') as out_obj:
			header = "participant,task,attempt,method,Rvel,Rcup,Rtable,Rvel*,Rcup*,Rtable*,RvD,RcD,RtD,Force,iactT,weight\n"
			out_obj.write(header)
			# participant ID can take values 0 - 9
			for ID in obj_metrics.keys():
				for task in obj_metrics[ID]:
					# trial can take values 1 or 2
					for trial in obj_metrics[ID][task]:
						for method in obj_metrics[ID][task][trial]:
							row = "P"+str(ID)+",T"+str(task)+","+str(trial)+","+method
							out_obj.write(row)
							for num in obj_metrics[ID][task][trial][method]:
								out_obj.write(","+str(num))
							out_obj.write('\n')
		out_obj.close()

		"""
		# write subjective metrics
		with open(filepath_subj, 'w') as out_subj:
			header = "participant,method,age,gender,Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q9,Q10\n"
			out_subj.write(header)
			for ID in subj_metrics.keys():
				for method in subj_metrics[ID]:
					row = "P"+str(ID)+","+method
					out_subj.write(row)
					for num in subj_metrics[ID][method]:
						out_subj.write(","+str(num))
			out_subj.write('\n')
		out_subj.close()
		"""

# ------ Creates large dictionary of all relevant statistics -----------#

def compute_obj_metrics():
	"""
	Computes the optimal reward, force, and total interaction time for all
	participants across all trials and all experimental conditions.
	"""
	# each participant does task 1 (ONE_FEAT), 2 (TWO_FEAT) 
	# and has attempt 1,2 with Method A (ALL), B (MAX) = 2*4*N
	# objective metrics: optimal reward, avg_force, weight_metric, total iact time = 4 
	# participant ID can take values 0 - 11

	effortData = data_io.parse_exp_data("force")
	trackedData = data_io.parse_exp_data("tracked")
	weightData = data_io.parse_exp_data("weights")

	obj_metrics = {}

	# compute effort metrics
	for ID in effortData.keys():
		for task in effortData[ID]:
			# trial can take values 1 or 2
			for trial in effortData[ID][task]:
				for method in effortData[ID][task][trial]:
					# sanity checks
						if ID not in obj_metrics:
							obj_metrics[ID] = {}
						if task not in obj_metrics[ID]:
							obj_metrics[ID][task] = {}
						if trial not in obj_metrics[ID][task]:
							obj_metrics[ID][task][trial] = {}
						if method not in obj_metrics[ID][task][trial]:
							# stores: Rvel,Rcup,Rtable,Rvel*,Rcup*,Rtable*,RvD,RcD,RtD,Force,iactT,weight
							obj_metrics[ID][task][trial][method] = np.array([0.0]*12)

						# --- Compute Effort & Interact Time Metrics ---#
						data = effortData[ID][task][trial][method]
						effort = compute_effort(data,ID)
						iactT = compute_iactT(data,ID)
						obj_metrics[ID][task][trial][method][9] = effort
						obj_metrics[ID][task][trial][method][10] = iactT

						# --- Compute Weight Metrics ---#
						wdata = weightData[ID][task][trial][method]
						weight_metric = compute_weight(wdata,task)
						obj_metrics[ID][task][trial][method][11] = weight_metric

	# compute tracked trajectory metrics
	for ID in trackedData.keys():
		for task in trackedData[ID]:
			# compute optimal reward
			(Rvel_opt, Rcup_opt, Rtable_opt) = get_optimalReward(task)
			for trial in trackedData[ID][task]:
				for method in trackedData[ID][task][trial]:
					# --- Compute Reward ---#
					if method == "A":
						fmethod = ALL
					elif method == "B":
						fmethod = MAX
					plan = trajopt_planner.Planner(EXP_TASK, demo=False, featMethod=fmethod, numFeat=task)	
					data = trackedData[ID][task][trial][method]
					# data is: [[time1, j1, j2, j3, ... j7], [timeN, j1, j2, j3, ... j7]]
					(Rvel, Rcup, Rtable) = compute_reward(data, plan)
					# --- Store metrics ---#
					obj_metrics[ID][task][trial][method][0] = Rvel
					obj_metrics[ID][task][trial][method][1] = Rcup
					obj_metrics[ID][task][trial][method][2] = Rtable
					obj_metrics[ID][task][trial][method][3] = Rvel_opt
					obj_metrics[ID][task][trial][method][4] = Rcup_opt
					obj_metrics[ID][task][trial][method][5] = Rtable_opt
					obj_metrics[ID][task][trial][method][6] = Rvel_opt - Rvel
					obj_metrics[ID][task][trial][method][7] = Rcup_opt - Rcup
					obj_metrics[ID][task][trial][method][8] = Rtable_opt - Rtable
					plan.kill_planner()

	return obj_metrics


# ------ Utils ------ #

def get_optimalReward(task):
	"""
	Returns the optimal reward for given task. Precomputed. 
	"""
	if task == ONE_FEAT: # one feature task
		return (0.0252667811742, 67.9221456532, 41.5027278248)  
	elif task == TWO_FEAT: # two feature task
		return (0.0521496254811, 117.782620548, 61.2207211307)  
	else:
		print "wrong task number!"
		return (0,0,0)

def compute_optimalReward(task):
	"""
	Computes optimal reward from scratch, given task. 
	"""
	T = 20.0
	weights = [0.0,0.0]
	if task == ONE_FEAT:
		weights = [0.0,1.0]
	elif task == TWO_FEAT:
		weights = [1.0,1.0]

	if task == TWO_FEAT:
		# if two features are wrong, initialize the starting config badly (tilted cup)
		pick = pick_basic
		pick[-1] = 200.0
	else:
		pick = pick_basic 
	place = place_lower

	start = np.array(pick)*(math.pi/180.0)
	goal = np.array(place)*(math.pi/180.0)

	print "computing optimal reward"
	plan = trajopt_planner.Planner(EXP_TASK, demo=False, featMethod="ALL", numFeat=task)
	# choose 0.1 as step size to match real traj
	plan.replan(start, goal, weights, 0.0, T, 0.1, seed=None)	
	# use the upsampled traj from planner
	r = plan.featurize(plan.waypts)
	Rvel = r[0]
	Rfeat1 = np.sum(r[1])
	Rfeat2 = np.sum(r[2])
	print Rvel, Rfeat1, Rfeat2
	plan.kill_planner()
	return (Rvel, Rfeat1, Rfeat2)

def compute_effort(data,ID):
	"""
	Given one participant's force measurements for one trial of one experiment
	computes the total interaction time
	-----
	IMPT NOTE: 
	Participants with ID 0-7 had a script where it accidentally recorded the same
	interaction measurement 2 times! If you are analyzing one of those participants
	just count every other measurement. 
	"""
	# get only the data (no timestamps)
	edata = data[:,1:8]
	effort = 0.0
	if ID in range(0,8):
		incr = 2
	else:
		incr = 1
	for t in range(0,len(edata),incr):
		joint = edata[t]
		total = np.sum(np.abs(joint))
		effort += total

	return effort

def compute_iactT(data, ID):
	"""
	Given one participant's force measurements for one trial of one experiment
	computes the total interaction time	
	-----
	IMPT NOTE: 
	Participants with ID 0-7 had a script where it accidentally recorded the same
	interaction measurement 2 times! If you are analyzing one of those participants
	just count every other measurement. 
	"""
	time = data[:,0:1]
	if ID in range(0,8):
		count = len(time)/2
	else:
		count = len(time)
	# get only the timestamps
	totalT = count*0.1

	return totalT

def compute_reward(data, planner):
	"""
	Given one participant's tracked trajectory for one trial of one experiment
	computes the rewards from this trajectory
	"""
	# get only waypt data (no timestamps)
	waypts = data[:,1:8]

	r = planner.featurize(waypts)
	Rvel = r[0]
	Rcup = np.sum(r[1])
	Rtable = np.sum(r[2])
	#print "Rvel:" + str(Rvel)
	#print "Rcup:" + str(Rcup)
	return (Rvel, Rcup, Rtable)

def compute_weight(data, task):
	"""
	Given the weight data and the task that it was collected for, 
	computes the dot product between the weight at each time point
	and the ideal weight and averages the total.
	"""
	if task == ONE_FEAT:
		ideal_w = np.array([0.0,1.0])
	elif task == TWO_FEAT:
		ideal_w = np.array([1.0,1.0])

	timestamp = data[:,0:1]
	weights = data[:,1:len(data)+1]
	total = 0.0

	for t in range(len(timestamp)):
		#w_prev = np.array([weights[t-1]])
		# get weight at current time step
		#w_curr = np.array(weights[t])
		#total += np.linalg.norm(w_curr - w_prev)

		# get weight at current time step
		w = np.array(weights[t])
		# do dot product with ideal_w
		d = np.dot(w,ideal_w)
		total += d

	return total/len(timestamp)


if __name__ == '__main__':
	#compute_obj_metrics()
	
	filename = "metrics"
	save_parsed_data(filename, csvData=True, pickleData=True)
	
