import numpy as np
from numpy import linalg
from numpy import linspace
from matplotlib import rc
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import math
import logging
import copy
import os
import pickle
import data_io
import trajopt_planner
import time

import openrave_utils
from openrave_utils import *

# UPDATE THIS WITH THE CORRECT NUMBER OF PEOPLE
NUM_PPL = 12

pick_basic = [104.2, 151.6, 183.8, 101.8, 224.2, 216.9, 310.8]
pick_shelf = [210.8, 241.0, 209.2, 97.8, 316.8, 91.9, 322.8]
place_lower = [210.8, 101.6, 192.0, 114.7, 222.2, 246.1, 322.0]
place_higher = [210.5,118.5,192.5,105.4,229.15,245.47,316.4]

EXP_TASK = 2


def get_pickled_metrics(filename):
	"""
	Parses cleaned pickle file with metrics per participant.
	filename can be "metrics_obj.p" or "metrics_subj.p"
	"""

	here = os.path.dirname(os.path.realpath(__file__))
	subdir = "/data/experimental/"
	datapath = here + subdir + filename

	metrics = pickle.load( open( datapath, "rb" ) )
	return metrics

# ---- Plotting Functionality ---- #

def plotting(avgA, avgB, stdA, stdB, xlabel, ylabel, title, maxY, twostar=False, pval=[0.01,0.01]):
	ind = np.arange(2)  # the x locations for the groups
	width = 0.45       # the width of the bars
	offset = 0.15

	# colors
	blackC = "black"	#(214/255., 39/255., 40/255.)
	greyC = "grey"		#(44/255., 160/255., 44/255.)
	blueC = "#4BABC5" 	#(31/255., 119/255., 180/255.)
	orangeC = "#F79545" #(255/255., 127/255., 14/255.)

	# fonts
	rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
	rc('text', usetex=True)

	fig, ax = plt.subplots()


	rectsA = ax.bar(ind+offset, avgA, width, color=greyC, yerr=stdA, ecolor='k', edgecolor='#272727',linewidth=0.5,error_kw=dict(ecolor='black', lw=2, capsize=0, capthick=0))
	rectsB = ax.bar(ind+width+offset, avgB, width, color=orangeC, yerr=stdB, ecolor='k',linewidth=0.5, edgecolor='#272727',error_kw=dict(ecolor='black', lw=2, capsize=0, capthick=0))


	def autolabel(rects):
		"""
		Attach a text label above each bar displaying its height
		"""
		for rect in rects:
			height = rect.get_height()
			ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,'%.2f' % 
					height,ha='center', va='bottom', fontsize=15)

	def autolabel_star(rects, std, p, second=False, twostar=False):
		"""
		Attach a text label above each bar displaying its height
		"""
		for i in range(0,len(rects),2):
			height1 = rects[i].get_height()
			height2 = rects[i+1].get_height()
			height = max(height1,height2)

			stdh = max(std[i],std[i+1])*1.5
			print stdh
			x = (rects[i].get_x() + rects[i].get_width())			
			if second:
				x = (rects[i].get_x() + rects[i].get_width())*1.5
			y = stdh+height+2

			widthB = "widthB="+str((rects[i].get_width()+rects[i+1].get_width())*7)

			if twostar:
				ax.annotate(r'\textbf{**}', xy=(x, y), xytext=(x, y), xycoords='data', fontsize=25, ha='center', va='bottom',arrowprops=dict(arrowstyle='-[, '+widthB+', lengthB=1.2', lw=1.5))
			else:			
				ax.annotate(r'\textbf{*}', xy=(x, y), xytext=(x, y), xycoords='data', fontsize=25, ha='center', va='bottom',arrowprops=dict(arrowstyle='-[, '+widthB+', lengthB=1.2', lw=1.5))

			ax.text(x,y+1,r"p$<$"+str(p[i]),ha='center', va='bottom', fontsize=30)

	ptime = []
	peffort = []
	autolabel_star(rectsA,stdA,pval)
	autolabel_star(rectsB,stdB,pval, second=True)
	#autolabel(rectsB)

	# add some text for labels, title and axes ticks
	ax.set_ylabel(r'\textbf{'+ylabel+'}',fontsize=50,labelpad=15)
	ax.set_xlabel(r'\textbf{'+xlabel+'}',fontsize=50,labelpad=15)
	
	plt.text(0.5, 1.08, r'\textbf{'+title+'}',
			 horizontalalignment='center',
			 fontsize=55,
			 transform = ax.transAxes)
	
	ax.set_xticks(ind+width+offset)

	xlabels = ["Table","Table + Cup"] #["T"+str(t+1) for t in range(3)]
	ax.set_xticklabels(xlabels,10,fontsize=50)

	# remove the plot frame lines
	ax.spines["top"].set_visible(False)    
	ax.spines["right"].set_visible(False)      
	
	# set max y-limit 
	ax.set_ylim([0,maxY])
	ax.tick_params(labelsize=50)

	# set padding for x and y tick labels
	ax.tick_params(direction='out', pad=2)

	# ensure that the axis ticks only show up on left of the plot.  
	ax.xaxis.set_ticks_position('none') 
	ax.yaxis.set_ticks_position('none') 		

	
	leg = ax.legend((rectsA[0], rectsB[0]), (r'\textbf{All-at-Once}', r'\textbf{One-at-a-Time}'), fontsize=40)

	leg.get_frame().set_linewidth(0.0)
	plt.show()

	return fig

def plotting_overtime(methodA, methodB, stdevA, stdevB):
	"""
	plots the metric over time for each method
	"""

def augment_weights(time, weights):
	"""
	Augments the weight data with 0.1 sec timesteps
	"""
	#print "time: " + str(time)
	cup_w = weights[:,0]
	table_w = weights[:,1]
	#print "cup_w: " + str(cup_w)
	#print "table_w: " + str(table_w)

	aug_time = [0.0]*200 # traj is 20 sec, sampling at 0.1 sec
	aug_cup = [0.0]*200
	aug_table = [0.0]*200

	aug_idx = 0
	idx = 0
	prev_cup = 0.0
	prev_table = 0.0
	times = np.arange(0.1, 20.0, 0.1)
	for t in times:
		aug_time[aug_idx] = t
		#clipped_t = round(time[idx][0],1)
		if idx < len(cup_w) and np.isclose(round(time[idx][0],1), t, rtol=1e-05, atol=1e-08, equal_nan=False):
			aug_cup[aug_idx] = cup_w[idx]
			aug_table[aug_idx] = table_w[idx]
			prev_cup = cup_w[idx]
			prev_table = table_w[idx]
			idx += 1
		else:
			aug_cup[aug_idx] = prev_cup
			aug_table[aug_idx] = prev_table
		aug_idx += 1

	aug_time[-1] = 20.0
	aug_cup[-1] = prev_cup
	aug_table[-1] = prev_table
	return (aug_time, aug_cup, aug_table)

def plot_weights(task, saveFig=False):
	"""
	For the current task, makes 4 plots of all the weight updates over time
	for all the participants. 
	Two sets of plots: 
		- left 2 are for Update ALL (top = cup weight, bottom = table)
		- right 2 are for Update ONE (top = cup weight, bottom = table)
	"""
	weightData = data_io.parse_exp_data("weights")

	f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')

	ax1.set_title('Update ALL')
	ax2.set_title('Update ONE')
	ax3.set_xlabel('time (s)')
	ax4.set_xlabel('time (s)')
	ax1.set_ylabel('weight')
	ax3.set_ylabel('weight')

	ax1.set_ylim([-1.5, 1.5])
	ax2.set_ylim([-1.5, 1.5])
	ax3.set_ylim([-1.5, 1.5])
	ax4.set_ylim([-1.5, 1.5])

	ax1.set_xlim([0, 20])
	ax2.set_xlim([0, 20])
	ax3.set_xlim([0, 20])
	ax4.set_xlim([0, 20])

	#ax2.set_title('Task 2: Table+Cup')

	greyC = "grey"		#(44/255., 160/255., 44/255.)
	blueC = "#4BABC5" 	#(31/255., 119/255., 180/255.)
	orangeC = "#F79545" #(255/255., 127/255., 14/255.)

	trial = 2

	if task == 1:
		f.suptitle("Task 1: Correct Distance to Table",fontsize=20)
	else:
		f.suptitle("Task 2: Correct Distance to Table + Cup Orientation",fontsize=20)
	Acount = 0
	Bcount = 0

	for ID in weightData.keys():
		#for task in weightData[ID]:
		# trial can take values 1 or 2
		#for trial in weightData[ID][task]:
		for method in weightData[ID][task][trial]:
			
			data = weightData[ID][task][trial][method]
			timestamp = data[:,0:1]
			weights = data[:,1:len(data)+1]

			(aug_time, aug_cup, aug_table) = augment_weights(timestamp, weights)

			if method == "A":
				if Acount != 0:
					ax1.plot(aug_time,aug_cup,linewidth=4.0, color=greyC)
					ax3.plot(aug_time,aug_table,linewidth=4.0, color='k')
				else:
					ax1.plot(aug_time,aug_cup,linewidth=4.0, color=greyC, label='Cup')
					ax3.plot(aug_time,aug_table,linewidth=4.0, color='k', label='Table')
					Acount += 1
					ax1.axhline(y=0, color='k', linestyle='-')
					ax1.legend()
					ax3.axhline(y=0, color='k', linestyle='-')
					ax3.legend()
			elif method == "B":
				if Bcount != 0:
					ax2.plot(aug_time,aug_cup,linewidth=4.0, color=orangeC)
					ax4.plot(aug_time,aug_table,linewidth=4.0, color='r')
				else:
					ax2.plot(aug_time,aug_cup,linewidth=4.0, color=orangeC, label='Cup')
					ax4.plot(aug_time,aug_table,linewidth=4.0, color='r', label='Table')
					Bcount += 1
					ax2.axhline(y=0, color='k', linestyle='-')
					ax2.legend()				
					ax4.axhline(y=0, color='k', linestyle='-')
					ax4.legend()
				
	plt.show()

	if saveFig:
		here = os.path.dirname(os.path.realpath(__file__))
		subdir = "/data/experimental/"
		datapath = here + subdir
		f.savefig(datapath+"task"+str(task)+"Weights.pdf", bbox_inches="tight")
		print "Saved weights figure."

def plot_cupTableDiffFinal(saveFig=False):
	"""
	Produces two plots, one for the cupDiff and one for the tableDiff metric
	"""
	filename = "metrics_obj.p"
	obj = get_pickled_metrics(filename)

	# for keeping average of each feature, for each method, and for each task
	cupDiffAvg = np.array([[0.0, 0.0],[0.0,0.0]]) # [method all --> [avg for task 1, task 2], method one --> [avg for task 1, task 2]]
	tableDiffAvg = np.array([[0.0, 0.0],[0.0,0.0]]) 

	# for computing stddev 
	pplCupALL = np.array([[[0.0,0.0],[0.0,0.0]],[[0.0,0.0],[0.0,0.0]]]*NUM_PPL) # trial 1 --> [task 1, task 2], trial 2 --> [task 1, task 2]
	pplCupONE = np.array([[[0.0,0.0],[0.0,0.0]],[[0.0,0.0],[0.0,0.0]]]*NUM_PPL)
	pplTableALL = np.array([[[0.0,0.0],[0.0,0.0]],[[0.0,0.0],[0.0,0.0]]]*NUM_PPL)
	pplTableONE = np.array([[[0.0,0.0],[0.0,0.0]],[[0.0,0.0],[0.0,0.0]]]*NUM_PPL)

	stdCup = np.array([[0.0,0.0]]*2)
	stdTable = np.array([[0.0,0.0]]*2)

	for ID in obj.keys():
		for task in obj[ID]:
			for trial in [1,2]:
				cup_all = obj[ID][task][trial]["A"][17]
				cup_one = obj[ID][task][trial]["B"][17]
				table_all = obj[ID][task][trial]["A"][18]
				table_one = obj[ID][task][trial]["B"][18]

				cupDiffAvg[0][task-1] += cup_all
				cupDiffAvg[1][task-1] += cup_one

				tableDiffAvg[0][task-1] += table_all
				tableDiffAvg[1][task-1] += table_one

				pplCupALL[ID][trial-1][task-1] = cup_all
				pplCupONE[ID][trial-1][task-1] = cup_one
				pplTableALL[ID][trial-1][task-1] = table_all
				pplTableONE[ID][trial-1][task-1] = table_one

	#print "pplCup: " + str(pplCup)
	# average by number of participants
	for method in range(2):
		for task in range(2):
			cupDiffAvg[method][task] /= NUM_PPL*2 # because 2 trials
			tableDiffAvg[method][task] /= NUM_PPL*2
			
			#c = np.reshape(pplCup[:,method,:,task], NUM_PPL)
			if method == 0: # all method
				stdCup[method][task] = np.std(pplCupALL[:,:,task])/np.sqrt(NUM_PPL*2*2) # because 2 tasks, 2 trials (per method)
				stdTable[method][task] = np.std(pplTableALL[:,:,task])/np.sqrt(NUM_PPL*2*2)
			else: # one method
				stdCup[method][task] = np.std(pplCupONE[:,:,task])/np.sqrt(NUM_PPL*2*2)
				stdTable[method][task] = np.std(pplTableONE[:,:,task])/np.sqrt(NUM_PPL*2*2)
		
	print stdCup
	xlabel = "Task"
	ylabel = r"CupDiffFinal"
	title = r"\textit{CupDiffFinal}: Cup Feature Difference from Ideal"	
	maxY = 50.0
	fig = plotting(cupDiffAvg[0], cupDiffAvg[1], stdCup[0], stdCup[1], xlabel, ylabel, title, maxY, pval=[0.01,0.01])

	if saveFig:
		here = os.path.dirname(os.path.realpath(__file__))
		subdir = "/data/experimental/"
		datapath = here + subdir
		fig.savefig(datapath+"cupDiff.pdf", bbox_inches="tight")
		print "Saved cupDiff figure."

	xlabel = "Task"
	ylabel = r"TableDiffFinal"
	title = r"\textit{TableDiffFinal}: Table Feature Difference from Ideal"	
	maxY = 20.0
	fig = plotting(tableDiffAvg[0], tableDiffAvg[1], stdTable[0], stdTable[1], xlabel, ylabel, title, maxY, twostar=True, pval=[0.001,0.001])

	if saveFig:
		here = os.path.dirname(os.path.realpath(__file__))
		subdir = "/data/experimental/"
		datapath = here + subdir
		fig.savefig(datapath+"tableDiff.pdf", bbox_inches="tight")
		print "Saved tableDiff figure."

def plot_dotOverTime(saveFig=False):
	"""
	Produces plot over time of the dot product
	"""
	weightData = data_io.parse_exp_data("weights")

	times = np.arange(0.0, 20.0, 0.1)
	print "times: " + str(times)
	dot_ALL = {}
	dot_ONE = {}
	for task in [1,2]:
		dot_ALL[task] = {}
		dot_ONE[task] = {}
		for ID in range(NUM_PPL):
			dot_ALL[task][ID] = {}
			dot_ONE[task][ID] = {}
			#for t in times:
			#	dot_ALL[task][ID][t] = None#[0.0,0.0] #trial 1, trial 2
			#	dot_ONE[task][ID][t] = None#[0.0,0.0] 

	aug_times = None
	# get weight data
	for ID in weightData.keys():
		for task in weightData[ID]:
			if task == 1:
				ideal_w = np.array([0.0,1.0])
			elif task == 2:
				ideal_w = np.array([1.0,1.0])

			#for trial in weightData[ID][task]:
			for method in ["A", "B"]:
				data1 = weightData[ID][task][1][method]
				data2 = weightData[ID][task][2][method]

				timestamp1 = data1[:,0:1]
				timestamp2 = data2[:,0:1]
				weights1 = data1[:,1:len(data1)+1]
				weights2 = data2[:,1:len(data2)+1]

				(aug_time1, aug_cup1, aug_table1) = augment_weights(timestamp1, weights1)
				(aug_time2, aug_cup2, aug_table2) = augment_weights(timestamp2, weights2)
				print "aug_time1: " + str(aug_time1)
				print "aug_time2: " + str(aug_time2)
				for t in range(len(aug_time1)):
					if aug_time1[t] not in dot_ALL[task][ID]:
						dot_ALL[task][ID][aug_time1[t]] = [0.0,0.0]
					if aug_time1[t] not in dot_ONE[task][ID]:
						dot_ONE[task][ID][aug_time1[t]] = [0.0,0.0]

					w1 = np.array([aug_cup1[t],aug_table1[t]]) 
					w2 = np.array([aug_cup2[t],aug_table2[t]]) 
					curr_dot = [np.dot(w1,ideal_w), np.dot(w2,ideal_w)] # dot prod for trial 1 and trial 2
					if method == "A":
						#print "w1: " + str(w1)
						#print "w2: " + str(w2)
						dot_ALL[task][ID][aug_time1[t]] = curr_dot
						print "dot all: " + str(curr_dot)
					else:
						dot_ONE[task][ID][aug_time1[t]] = curr_dot
						print "dot one: " + str(curr_dot)

				aug_times = aug_time1

	# now do the averaging over time across each participant
	avgALLT1 = [0.0]*len(times)
	avgONET1 = [0.0]*len(times)
	avgALLT2 = [0.0]*len(times)
	avgONET2 = [0.0]*len(times)

	idx = 0
	for t in aug_times:
		aAllt1 = 0.0
		aOnet1 = 0.0
		aAllt2 = 0.0
		aOnet2 = 0.0
		for ID in range(NUM_PPL):
			print "in avg: dot all: " + str(dot_ALL[1][ID][t])
			aAllt1 += dot_ALL[1][ID][t][0] + dot_ALL[1][ID][t][1]
			aOnet1 += dot_ONE[1][ID][t][0] + dot_ONE[1][ID][t][1]
			aAllt2 += dot_ALL[2][ID][t][0] + dot_ALL[2][ID][t][1]
			aOnet2 += dot_ONE[2][ID][t][0] + dot_ONE[2][ID][t][1]
		print aAllt1
		avgALLT1[idx] = aAllt1/(NUM_PPL*2)
		avgONET1[idx] = aOnet1/(NUM_PPL*2)
		avgALLT2[idx] = aAllt2/(NUM_PPL*2)
		avgONET2[idx] = aOnet2/(NUM_PPL*2)
		idx += 1

	# colors
	blackC = "black"	#(214/255., 39/255., 40/255.)
	greyC = "grey"		#(44/255., 160/255., 44/255.)
	blueC = "#4BABC5" 	#(31/255., 119/255., 180/255.)
	orangeC = "#F79545" #(255/255., 127/255., 14/255.)

	# fonts
	rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
	rc('text', usetex=True)
	
	fig, ax = plt.subplots()

	plt.text(0.5, 1.08, r'\textbf{Task 1 \textit{DotAvg} Over Duration of Trajectory}',
			 horizontalalignment='center',
			 fontsize=35,
			 transform = ax.transAxes)


	#ax.set_xticks(ind+width+offset)

	#xlabels = "Time (s)" #["T"+str(t+1) for t in range(3)]
	#ax.set_xticklabels(xlabels,10,fontsize=50)

	plt.ylabel('DotAvg',fontsize=50)
	plt.xlabel('Time (s)',fontsize=50)

	# remove the plot frame lines
	ax.spines["top"].set_visible(False)    
	ax.spines["right"].set_visible(False)      
	
	ax.tick_params(labelsize=50)
	# set padding for x and y tick labels
	ax.tick_params(direction='out', pad=2)

	# ensure that the axis ticks only show up on left of the plot.  
	ax.xaxis.set_ticks_position('none') 
	ax.yaxis.set_ticks_position('none') 		

	# plot the average dot product over time
	allT1 = ax.plot(times, avgALLT1, '-o', color=greyC, label=r'\textbf{All-at-Once}')
	oneT1 = ax.plot(times, avgONET1, '-o', color=orangeC, label=r'\textbf{One-at-a-Time}')

	# TODO THIS ISN'T DONE
	allT1.fill_between(x, y-error, y+error)
	oneT1.fill_between(x, y-error, y+error)

	leg = ax.legend() #ax.legend((allT1, oneT1), (r'\textbf{All-at-Once}', r'\textbf{One-at-a-Time}'), fontsize=40)

	plt.show()

	if saveFig:
		here = os.path.dirname(os.path.realpath(__file__))
		subdir = "/data/experimental/"
		datapath = here + subdir
		fig.savefig(datapath+"dotAvgTime.pdf", bbox_inches="tight")
		print "Saved dotAvgTime figure."

#--------- OprenRAVE plotting --------#

def plot_taskOpenrave(task):
	"""
	Plots the default and ideal trajectory for the task.
	"""

	blackC = [0.2,0.2,0.2] #[0,0,0] 	
	greyC =  [0.5,0.5,0.5]		
	blueC = [75/255.0,171/255.0,197/255.]
	orangeC = [247/255.,149/255.,69/255.0] 

	# total traj time
	T = 20.0

	ideal_w = [0.0,0.0]
	if task == 1:
		ideal_w = [0.0,1.0]
	elif task == 2:
		ideal_w = [1.0,1.0]

	if task == 2:
		# if two features are wrong, initialize the starting config badly (tilted cup)
		pick = copy.copy(pick_basic)
		pick[-1] = 200.0
	else:
		pick = copy.copy(pick_basic) 
	place = copy.copy(place_lower)

	start = np.array(pick)*(math.pi/180.0)
	goal = np.array(place)*(math.pi/180.0)

	plan = trajopt_planner.Planner(EXP_TASK, demo=False, featMethod="MAX", numFeat=task)

	# --- ideal traj --- #
	# choose 0.1 as step size to match real traj
	plan.replan(start, goal, ideal_w, 0.0, T, 0.1, seed=None)	
	# plot ideal trajectory
	plotCupTraj(plan.env,plan.robot,plan.bodies,plan.waypts,color=blueC,increment=20)

	# --- default traj --- #
	default_w = [0.0,0.0]
	# choose 0.1 as step size to match real traj
	plan.replan(start, goal, default_w, 0.0, T, 0.1, seed=None)	
	# plot ideal trajectory
	plotCupTraj(plan.env,plan.robot,plan.bodies,plan.waypts,color=blackC,increment=20)

	time.sleep(30)

if __name__ == '__main__':

	# --- for plotting trajectory default and ideal ---- #
	#task = 1
	#plot_taskOpenrave(task)

	#plot_cupTableDiffFinal(True)
	plot_dotOverTime(True)
