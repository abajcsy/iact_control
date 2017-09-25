import numpy as np
from numpy import linalg
from numpy import linspace
from matplotlib import rc
import matplotlib.pyplot as plt
import time
import scipy
import math
import pickle
import logging
import copy

import csv
import os

import openrave_utils
from openrave_utils import *

import experiment_utils
from experiment_utils import *

import trajopt_planner
from trajopt_planner import *

from scipy import stats

pick_basic = [104.2, 151.6, 183.8, 101.8, 224.2, 216.9, 310.8]
pick_shelf = [210.8, 241.0, 209.2, 97.8, 316.8, 91.9, 322.8]
place_lower = [210.8, 101.6, 192.0, 114.7, 222.2, 246.1, 322.0]
place_higher = [210.5,118.5,192.5,105.4,229.15,245.47,316.4]

ONE_FEAT = 1					# experiment where human has to correct only one feature (out of three)
TWO_FEAT = 2					# experiment where human has to correc two features

ALL = "ALL" 							# updates all features
MAX = "MAX"								# updates only feature that changed the most
LIKELY = "LIKELY"						# updates the most likely feature 

FAM_TASK = 1
EXP_TASK = 2

def compute_effort(data):
	"""
	Given one participant's force measurements for one trial of one experiment
	computes the total interaction time
	"""
	# get only the data (no timestamps)
	edata = data[1:8]
	(h,w) = np.shape(edata)
	effort = 0.0
	for t in range(w):
		joint = edata[:,t]
		total = np.sum(np.abs(joint))
		effort += total

	return effort

def compute_iactT(data):
	"""
	Given one participant's force measurements for one trial of one experiment
	computes the total interaction time	
	"""
	# get only the timestamps
	time = data[0]
	prevT = time[0]

	totalT = len(time)*0.1

	return totalT

def compute_reward(data, planner):
	"""
	Given one participant's tracked trajectory for one trial of one experiment
	computes the reward from this trajectory
	"""
	# get only waypt data (no timestamps)
	waypts = data[1:8].T
	
	r = planner.featurize(waypts)
	Rvel = r[0]
	Rfeat = np.sum(r[1])
	print "Rvel:" + str(Rvel)
	print "Rfeat:" + str(Rfeat)
	return (Rvel, Rfeat)

def compute_optimalReward(task, precomputed=True):

	#TODO THIS WHOLE FUNCTION NEEDS TO BE CHANGED 

	"""
	Compute the optimal feature values for given task.
	Precomputed=True means it uses old values, set to False if want to recompute.
	"""

	if precomputed:
		if task == ONE_FEAT: # one feature task
			return (1.6849658205633371, 1.030981635333591, 0.39520399865015937)  #return (0.10292734749900004, 37.699161124896193)
		elif task == TWO_FEAT: # two feature task
			return (3.4797942681227552, 1.1488477888843827, 0.6889346436935464)  # return (0.029572791376999984, 37.420166769225823)
	else:
		T = 20.0
		weights = [0.0,0.0]
		if task == ONE_FEAT:
			weights[0] = [0.0,1.0]
		elif task == TWO_FEAT:
			weights = [1.0,1.0]

		if task == TWO_FEAT:
			# if two features are wrong, initialize the starting config badly (tilted cup)
			pick = pick_basic
			pick[-1] = 200.0
		else:
			pick = pick_basic 
		place = place_lower

		startRad = np.array(pick)*(math.pi/180.0)
		goalRad = np.array(place)*(math.pi/180.0)
		start = startRad
		goal = goalRad
	
		filename = None
		if task == ONE_FEAT:
			filename = "/home/abajcsy/catkin_ws/src/iact_control/src/task1_opt.p"
		elif task == TWO_FEAT:
			filename = "/home/abajcsy/catkin_ws/src/iact_control/src/task2_opt.p"
		
		pickle.load( open( filename, "rb" ) )
		# get optimal waypts from file
		#waypts = get_opt_waypts(filename)

		print "computing optimal reward"
		plan = Planner(EXP_TASK, demo=False, featMethod="ALL", numFeat=task)
		r = plan.featurize(waypts)
		Rvel = r[0]
		Rfeat1 = np.sum(r[1])
		Rfeat2 = np.sum(r[2])
		print Rvel, Rfeat1, Rfeat2
		plan.kill_planner()
		return (Rvel, Rfeat1, Rfeat2)

def get_opt_waypts(filename):
	"""
	Reads the optimal waypoints from file
	"""
	# get ground truth for task 2 only!!!
	here = os.path.dirname(os.path.realpath(__file__))
	subdir = "/data/experimental/"
	datapath = here + subdir + filename
	firstline = True
	waypts = None
	with open(datapath, 'r') as f:
		methodData = [None]*8
		i = 0
		for line in f:
			# skip first line in tracked that has totalT
			if firstline:
				firstline = False
				continue
			values = line.split(',')
			final_values = [float(v) for v in values[1:len(values)]]
			methodData[i] = final_values
			i += 1
		data = np.array(methodData)
		waypts = data
	return waypts[1:8].T

def save_opt_waypts(task):
	"""
	Computes the optimal waypoints given a task and pickles file
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

	startRad = np.array(pick)*(math.pi/180.0)
	goalRad = np.array(place)*(math.pi/180.0)
	start = startRad
	goal = goalRad

	plan = Planner(EXP_TASK, demo=False, featMethod="ALL", numFeat=task)	
	print "made planner"
	print weights
	traj = plan.replan(start, goal, weights, 0.0, T, 0.5, seed=None)
	print traj	
	filename = "task" + str(task) + "_opt.p"
	pickle.dump(traj, open( filename, "wb" ) )
	r = plan.featurize(traj)
	Rvel = r[0]
	Rfeat1 = np.sum(r[1])
	Rfeat2 = np.sum(r[2])
	print Rvel, Rfeat1, Rfeat2
	plan.kill_planner()
	return (Rvel, Rfeat1, Rfeat2)

def compute_obj_metrics():
	"""
	Computes the optimal reward, force, and total interaction time for all
	participants across all trials and all experimental conditions.
	"""
	# each participant does task 1,2,3 and has attempt 1,2 with method A,B = 3*4*N
	# objective metrics include optimal reward, avg_force, total iact time = 3 

	parser = ExperimentUtils()
	effortData = parser.parse_data("force")
	trackedData = parser.parse_data("tracked")

	obj_metrics = {}

	# participant ID can take values 0 - 9
	for ID in effortData.keys():
		for task in effortData[ID]:
			# dont include the familiarization task (task can take values 1,2,3)
			if task != 0:
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
							# stores: Rv,Rf,Rv*,Rf*,RvD,RfD,Force,iactT
							obj_metrics[ID][task][trial][method] = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])

						# --- Compute Effort & Interact Time Metrics ---#
						data = effortData[ID][task][trial][method]
						effort = compute_effort(data)
						iactT = compute_iactT(data)
						obj_metrics[ID][task][trial][method][6] = effort
						obj_metrics[ID][task][trial][method][7] = iactT

	# 	NOTE: You have to turn off QtCoin viewer in openrave_utils initialize() function
	# 	for you to be able to call Planner through openrave multiple times in a row

	for ID in trackedData.keys():
		for task in trackedData[ID]:
			if task != 0:
				# compute optimal reward
				(Rvel_opt, Rfeat_opt) = compute_optimalReward(task, precomputed=True)	
				for trial in trackedData[ID][task]:
					for method in trackedData[ID][task][trial]:
						plan = Planner(EXP_TASK, demo=False, featMethod=method, numFeat=task)
						# --- Compute Reward ---#
						data = trackedData[ID][task][trial][method]
						(Rvel,Rfeat) = compute_reward(data, plan)
						obj_metrics[ID][task][trial][method][0] = Rvel
						obj_metrics[ID][task][trial][method][1] = Rfeat
						obj_metrics[ID][task][trial][method][2] = Rvel_opt
						obj_metrics[ID][task][trial][method][3] = Rfeat_opt
						obj_metrics[ID][task][trial][method][4] = Rvel_opt - Rvel
						obj_metrics[ID][task][trial][method][5] = Rfeat_opt - Rfeat
						plan.kill_planner()
	return obj_metrics

def compute_subj_metrics():
	"""
	Computes all subjective metric Likert data.
	"""
	# each participant does task 1,2,3 with method A,B = 3*2*N
	# likert data includes age, gender, Q1 - Q10 = 2+10

	# set up data structure
	subj_metrics = {}

	"""
	TODO: the ID range shold change depending 
		  on numbering scheme for experiment
	"""
	IDNums = [i for i in range(2,8)]

	for ID in IDNums:
		for method in ["A","B"]:
			# sanity checks
			if ID not in subj_metrics:
				subj_metrics[ID] = {}
			if method not in subj_metrics[ID]:
				subj_metrics[ID][method] = [None]*15

	here = os.path.dirname(os.path.realpath(__file__))
	subdir = "/data/experimental/"
	datapath = here + subdir + "Likert_Responses.csv"

	data = {}
	firstline = True
	with open(datapath, 'r') as f:
		for line in f:
			if firstline:
				firstline = False
				continue
			values = line.split(',')
			info = values[1:len(values)]
			print info
			ID = int(info[0])
			method = info[1]
			age = info[2]
			gender = info[3]
			tech_bg = info[4]
			# store age
			subj_metrics[ID][method][0] = age
			subj_metrics[ID][method][1] = gender
			# parse likert data (13 Q's in total)
			for i in range(5,18):
				subj_metrics[ID][method][i-3] = info[i]

	return subj_metrics

def	reorganize_data(filename):
	"""
	Restructures all data from participants into a single file for
	objective measures and subjective measures.
	"""
	obj_metrics = compute_obj_metrics()
	#subj_metrics = compute_subj_metrics() 		# TODO UNCOMMENT ME
		

	# write to file
	here = os.path.dirname(os.path.realpath(__file__))
	subdir = "/data/experimental/"
	filepath_obj = here + subdir + filename + "_obj.csv"
	#filepath_subj = here + subdir + filename + "_subj.csv"	# TODO UNCOMMENT ME

	# write objective metrics
	with open(filepath_obj, 'w') as out_obj:
		header = "participant,task,attempt,method,Rv,Rf,Rv*,Rf*,RvD,RfD,Force,iactT\n"
		out_obj.write(header)
		# participant ID can take values 0 - 9
		for ID in obj_metrics.keys():
			for task in obj_metrics[ID]:
				# dont include the familiarization task (task can take values 1,2,3)
				if task != 0:
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
		header = "participant,method,age,gender,Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q9,Q10,Q11,Q12,Q13\n"
		out_subj.write(header)
		for ID in subj_metrics.keys():
			for method in subj_metrics[ID]:
				row = "P"+str(ID)+","+method
				out_subj.write(row)
				for num in subj_metrics[ID][method]:
					out_subj.write(","+str(num))
				out_subj.write('\n')
			#out_subj.write('\n')
	out_subj.close()
	"""

def test():
	filename = "tracked111A1.csv"
	here = os.path.dirname(os.path.realpath(__file__))
	subdir = "/data/experimental/"
	datapath = here + subdir + filename

	firstline = True
	with open(datapath, 'r') as f:
		methodData = [None]*8
		i = 0
		for line in f:
			# skip first line in tracked that has totalT
			if firstline:
				firstline = False
				continue
			values = line.split(',')
			final_values = [float(v) for v in values[1:len(values)]]
			methodData[i] = final_values
			i += 1
		data = np.array(methodData)
		print data
		plan = Planner(EXP_TASK, demo=False, featMethod="ALL", numFeat=1)
		res = compute_reward(data, plan)
		print res


if __name__ == '__main__':
	
	#filename = "pilot_metrics"
	#reorganize_data(filename)

	experi = ExperimentUtils()
	#ID = 9
	task = TWO_FEAT
	#method = "B"
	#trial = 2
	res = save_opt_waypts(task)
	#experi.plot_trajDebug(plan)

	print "task" + str(task) + " opt: " + str(res)
	
