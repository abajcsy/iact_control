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

import experiment_utils
from experiment_utils import *

import trajopt_planner
from trajopt_planner import *

from scipy import stats

pick_table = [104.2,151.6,183.8,101.8,224.2,216.9,310.8]
pick_shelf = [210.8, 241.0, 209.2, 97.8, 316.8, 91.9, 322.8]
place_lower = [210.8, 101.6, 192.0, 114.7, 222.2, 246.1, 322.0]
place_higher = [233.0,132.4,200.5,137.8,248.7,243.2,264.8]

COFFEE_TASK = 1
TABLE_TASK = 2
LAPTOP_TASK = 3

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
		#NOTE: used to be 2-norm: norm = np.linalg.norm(joint)
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
	"""
	Compute the optimal feature values for given task.
	Precomputed=True means it uses old values, set to False if want to recompute.
	"""

	if precomputed:
		if task == 1: 
			return (0.10292734749900004, 37.699161124896193)
		elif task == 2:
			return (0.029572791376999984, 37.420166769225823)
		elif task == 3:
			return (0.040048660123999998, 10.354323653427837)
	else:
		T = 15.0
		weights = 0
		if task == TABLE_TASK or task == COFFEE_TASK:
			weights = 1
		elif task == LAPTOP_TASK:
			weights = 10

		# initialize start/goal based on task
		if task == COFFEE_TASK or task == HUMAN_TASK:
			pick = pick_shelf
			place = place_lower
		else:
			pick = pick_table
			place = place_higher
		
		startRad = np.array(pick)*(math.pi/180.0)
		goalRad = np.array(place)*(math.pi/180.0)
		start = startRad
		goal = goalRad
	
		filename = None
		if task == 1:
			filename = "task1.csv"
		elif task == 2:
			filename = "task2.csv"
		elif task == 3:
			filename = "task3.csv"
		
		# get optimal waypts from file
		waypts = get_opt_waypts(filename)

		print "computing optimal reward"
		plan = Planner(task, demo=False)
		r = plan.featurize(waypts)
		Rvel = r[0]
		Rfeat = np.sum(r[1])
		print Rvel, Rfeat
		plan.kill_planner()
		return (Rvel, Rfeat)

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

def get_opt_plan(task):
	"""
	Computes the optimal plan given a task
	"""
	print "in get opt plan"
	T = 15.0
	weights = 0
	if task == TABLE_TASK or task == COFFEE_TASK:
		weights = 1
	elif task == LAPTOP_TASK:
		weights = 10

	# initialize start/goal based on task 
	if task == COFFEE_TASK or task == HUMAN_TASK:
		pick = pick_shelf
	else:
		pick = pick_basic

	if task == LAPTOP_TASK:
		place = place_higher
	else:
		place = place_lower
		
	startRad = np.array(pick)*(math.pi/180.0)
	goalRad = np.array(place)*(math.pi/180.0)
	start = startRad
	goal = goalRad

	plan = Planner(task, demo=False)	
	plan.replan(start, goal, weights, 0.0, T, 0.1)
	print "in get opt plan"
	plan.kill_planner()
	return plan

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
				plan = Planner(task, demo=False)	
				for trial in trackedData[ID][task]:
					for method in trackedData[ID][task][trial]:

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
	subj_metrics = compute_subj_metrics()
		

	# write to file
	here = os.path.dirname(os.path.realpath(__file__))
	subdir = "/data/experimental/"
	filepath_obj = here + subdir + filename + "_obj.csv"
	filepath_subj = here + subdir + filename + "_subj.csv"

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
		plan = Planner(1, demo=False)
		res = compute_reward(data, plan)
		print res


if __name__ == '__main__':
	
	filename = "pilot_metrics"
	#reorganize_data(filename)

	#experi = ExperimentUtils()
	#ID = 9
	#task = 2
	#method = "B"
	#trial = 2
	#plan = get_opt_waypts(task)
	#experi.plot_trajDebug(plan)

	#print "task 1 opt: " + str(compute_optimalReward(1), precomputed=False)
	
