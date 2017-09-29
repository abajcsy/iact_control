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

import openrave_utils
from openrave_utils import *

# TODO UPDATE THIS WITH THE NUMBER OF PEOPLE
NUM_PPL = 9

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

def plotting(avgA, avgB, stdA, stdB, xlabel, ylabel, title, maxY, avgOpt=None):
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

	# plots with stdev
	if avgOpt is None:
		rectsA = ax.bar(ind+offset, avgA, width, color=greyC, yerr=stdA, ecolor='k', edgecolor='#272727',linewidth=0.5,error_kw=dict(ecolor='black', lw=2, capsize=0, capthick=0))
		rectsB = ax.bar(ind+width+offset, avgB, width, color=orangeC, yerr=stdB, ecolor='k',linewidth=0.5, edgecolor='#272727',error_kw=dict(ecolor='black', lw=2, capsize=0, capthick=0))
	else: 
		width = 0.25
		print avgA
		rectsA_cup = ax.bar(ind+offset, avgA[:,0], width, color=greyC, yerr=stdA, ecolor='k', edgecolor='#272727',linewidth=0.5,error_kw=dict(ecolor='black', lw=2, capsize=0, capthick=0))
		rectsA_table = ax.bar(ind+offset, avgA[:,1], width, color=greyC, yerr=stdA, ecolor='k', edgecolor='#272727',hatch="/",linewidth=0.5,error_kw=dict(ecolor='black', lw=2, capsize=0, capthick=0))
		rectsB_cup = ax.bar(ind+offset+width, avgB[:,0], width, color=orangeC, yerr=stdB, ecolor='k',linewidth=0.5, edgecolor='#272727',error_kw=dict(ecolor='black', lw=2, capsize=0, capthick=0))
		rectsB_table = ax.bar(ind+offset+width, avgB[:,1], width, color=orangeC, yerr=stdB, ecolor='k',linewidth=0.5, edgecolor='#272727',hatch="/",error_kw=dict(ecolor='black', lw=2, capsize=0, capthick=0))
		rectsOpt_cup = ax.bar(ind+offset+width*2, avgOpt[:,0], width, color=blueC,ecolor='k',linewidth=0.5, edgecolor='#272727',error_kw=dict(ecolor='black', lw=2, capsize=0, capthick=0))	
		rectsOpt_table = ax.bar(ind+offset+width*2, avgOpt[:,1], width, color=blueC,ecolor='k',linewidth=0.5, edgecolor='#272727',hatch="/",error_kw=dict(ecolor='black', lw=2, capsize=0, capthick=0))		

	def autolabel(rects):
		"""
		Attach a text label above each bar displaying its height
		"""
		for rect in rects:
			height = rect.get_height()
			ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,'%.2f' % 
					height,ha='center', va='bottom', fontsize=15)

	def autolabel_star(rects, std,p):
		"""
		Attach a text label above each bar displaying its height
		"""
		for i in range(len(rects)):
			height = rects[i].get_height()
			# cost:
			#x = rects[i].get_x() + rects[i].get_width()*1.5
			#y = std[i]+height+10

			# time:
			#x = rects[i].get_x() + rects[i].get_width()
			#y = std[i]+height+1

			# effort:
			#x = rects[i].get_x() + rects[i].get_width()
			#y = std[i]+height+50
			
			# time: x, y+0.2, widthB=2.0
			# cost: x-0.12, y+0.8, widthB=1.4
			# effort: x, y+12, widthB=2.0
			#ax.annotate(r'\textbf{*}', xy=(x, y), xytext=(x, y+12), xycoords='data', fontsize=25, ha='center', va='bottom',arrowprops=dict(arrowstyle='-[, widthB=2.0, lengthB=1.2', lw=1.5))
			# time: y+1.0
			# cost: y+6, x-0.12
			# effort: y+60
			#ax.text(x-0.12,y+6,r"p$<$"+str(p[i]),ha='center', va='bottom', fontsize=20)

	#ptime = []
	#peffort = []
	#pcost = [0.001,0.001,0.001]
	#autolabel_star(rectsA,stdA,pcost)
	#autolabel(rectsB)

	# add some text for labels, title and axes ticks
	ax.set_ylabel(r'\textbf{'+ylabel+'}',fontsize=30,labelpad=15)
	ax.set_xlabel(r'\textbf{'+xlabel+'}',fontsize=30,labelpad=15)
	
	plt.text(0.5, 1.08, r'\textbf{'+title+'}',
			 horizontalalignment='center',
			 fontsize=33,
			 transform = ax.transAxes)
	
	ax.set_xticks(ind+width+offset)
	if avgOpt is not None:
		ax.set_xticks(ind+width+width/2+offset)

	xlabels = ["Table","Table + Cup"]#["T"+str(t+1) for t in range(3)]
	ax.set_xticklabels(xlabels,10,fontsize=30)

	# remove the plot frame lines
	ax.spines["top"].set_visible(False)    
	ax.spines["right"].set_visible(False)      
	
	# set max y-limit 
	ax.set_ylim([0,maxY])
	ax.tick_params(labelsize=30)

	# set padding for x and y tick labels
	ax.tick_params(direction='out', pad=2)

	# ensure that the axis ticks only show up on left of the plot.  
	ax.xaxis.set_ticks_position('none') 
	ax.yaxis.set_ticks_position('none') 		

	if avgOpt is not None:		
		leg = ax.legend((rectsA_cup, rectsA_table, rectsB_cup, rectsB_table, rectsOpt_cup, rectsOpt_table), (r'\textbf{Update-All: cup}', r'\textbf{Update-ALL: table}', r'\textbf{Update-ONE: cup}', r'\textbf{Update-ONE: table}', r'\textbf{Desired: cup}', r'\textbf{Desired: table}'), fontsize=20)
		#leg = ax.legend((rectsA_T1, rectsA_T2, rectsB_T1, rectsB_T2, rectsOpt_T1, rectsOpt_T2), (r'\textbf{Update-All: Table}', r'\textbf{Update-All: Table+Cup}' r'\textbf{Update-One: Table}', r'\textbf{Update-One: Table+Cup}', r'\textbf{Desired: Table}', r'\textbf{Desired: Table+Cup}'), fontsize=20)
	else: 
		leg = ax.legend((rectsA[0], rectsB[0]), (r'\textbf{Update-All}', r'\textbf{Update-One}'), fontsize=23)

	leg.get_frame().set_linewidth(0.0)
	plt.show()

	return fig

def plot_taskEffort(saveFig=False):
	"""
	Takes all participant data files and produces bar chart
	comparing average force exerted by each participant for each task
	----
	saveFig 	if True, saves final plot
	"""
	filename = "metrics_obj.p"
	metrics = get_pickled_metrics(filename)
	# store avg for task 1,2
	sumA = [0.0,0.0]
	sumB = [0.0,0.0]

	pplA = [[0.0]*NUM_PPL, [0.0]*NUM_PPL]
	pplB = [[0.0]*NUM_PPL, [0.0]*NUM_PPL]
	for ID in metrics.keys():
		for task in metrics[ID]:
			trialAvgA = 0.0
			trialAvgB = 0.0
			for trial in metrics[ID][task]:
				trialAvgA += metrics[ID][task][trial]["A"][9]
				trialAvgB += metrics[ID][task][trial]["B"][9]
			trialAvgA /= 2.0
			trialAvgB /= 2.0
			sumA[task-1] += trialAvgA
			sumB[task-1] += trialAvgB

			pplA[task-1][ID] = trialAvgA
			pplB[task-1][ID] = trialAvgB
	avgA = [a/NUM_PPL for a in sumA]
	stdA = [np.std(pplA[0]), np.std(pplA[1])]
	avgB = [b/NUM_PPL for b in sumB]
	stdB = [np.std(pplB[0]), np.std(pplB[1])]

	# plot data
	xlabel = "Task"
	ylabel = "Total Effort (Nm)"
	title = "Average Total Human Effort"	
	maxY = 400	
	fig =plotting(avgA,avgB,stdA,stdB,xlabel,ylabel,title,maxY)

	if saveFig:
		here = os.path.dirname(os.path.realpath(__file__))
		subdir = "/data/experimental/"
		datapath = here + subdir
		fig.savefig(datapath+"taskEffort.pdf", bbox_inches="tight")
		print "Saved effort figure." 

def plot_taskEffortTime(saveFig=False):
	"""
	Plot average amount of interaction time for each task, for each method.
	"""
	filename = "metrics_obj.p"
	metrics = get_pickled_metrics(filename)
	# store avg for task 1,2
	sumA = [0.0,0.0]
	sumB = [0.0,0.0]

	pplA = [[0.0]*NUM_PPL, [0.0]*NUM_PPL]
	pplB = [[0.0]*NUM_PPL, [0.0]*NUM_PPL]
	for ID in metrics.keys():
		for task in metrics[ID]:
			trialAvgA = 0.0
			trialAvgB = 0.0
			for trial in metrics[ID][task]:
				trialAvgA += metrics[ID][task][trial]["A"][10]
				trialAvgB += metrics[ID][task][trial]["B"][10]
			trialAvgA /= 2.0
			trialAvgB /= 2.0
			sumA[task-1] += trialAvgA
			sumB[task-1] += trialAvgB

			pplA[task-1][ID] = trialAvgA
			pplB[task-1][ID] = trialAvgB
	avgA = [a/NUM_PPL for a in sumA]
	stdA = [np.std(pplA[0]), np.std(pplA[1])]
	avgB = [b/NUM_PPL for b in sumB]
	stdB = [np.std(pplB[0]), np.std(pplB[1])]

	# plot data
	xlabel = "Task"
	ylabel = "Interact Time (s)"
	title = "Average Total Interaction Time"	
	maxY = 5.0
	fig = plotting(avgA,avgB,stdA,stdB,xlabel,ylabel,title,maxY)

	if saveFig:
		here = os.path.dirname(os.path.realpath(__file__))
		subdir = "/data/experimental/"
		datapath = here + subdir
		fig.savefig(datapath+"taskTime.pdf", bbox_inches="tight")
		print "Saved time figure."

def plot_taskCost(saveFig=False):
	"""
	TODO Need to think about how to do this for 2 task costs
	"""
	filename = "metrics_obj.p"
	metrics = get_pickled_metrics(filename)
	# store avg for task 1,2

	# stores optimal cup,table costs for task 1 and task 2
	optimal = np.array([[0.0,0.0],[0.0,0.0]])
	sumA = np.array([[0.0,0.0],[0.0,0.0]])
	sumB = np.array([[0.0,0.0],[0.0,0.0]])

	pplA = np.array([[[0.0,0.0],[0.0,0.0]]]*NUM_PPL)
	pplB = np.array([[[0.0,0.0],[0.0,0.0]]]*NUM_PPL)

	for ID in metrics.keys():
		for task in metrics[ID]:
			trialAvgA = [0.0,0.0] # cup, table
			trialAvgB = [0.0,0.0] # cup, table
			for trial in metrics[ID][task]:
				trialAvgA[0] += metrics[ID][task][trial]["A"][1] #cup for method A
				trialAvgA[1] += metrics[ID][task][trial]["A"][2] #table for method B
				trialAvgB[0] += metrics[ID][task][trial]["B"][1] 
				trialAvgB[1] += metrics[ID][task][trial]["B"][2] 

			# get optimal cup and table cost for current task
			optimal[task-1][0] = metrics[ID][task][1]["B"][4] #cup
			optimal[task-1][1] = metrics[ID][task][1]["B"][5] #table

			trialAvgA[0] /= 2.0
			trialAvgA[1] /= 2.0
			trialAvgB[0] /= 2.0
			trialAvgB[1] /= 2.0
			sumA[task-1][0] += trialAvgA[0]
			sumA[task-1][1] += trialAvgA[1]
			sumB[task-1][0] += trialAvgB[0]
			sumB[task-1][1] += trialAvgB[1]

			pplA[ID][task-1][0] = trialAvgA[0]
			pplA[ID][task-1][1] = trialAvgA[1]
			pplB[ID][task-1][0] = trialAvgB[0]
			pplB[ID][task-1][1] = trialAvgB[1]

	avgA = np.array([[0.0,0.0]]*2)
	avgB = np.array([[0.0,0.0]]*2)
	stdA = np.array([[0.0,0.0]]*2)
	stdB = np.array([[0.0,0.0]]*2)

	avgA[0][0] = sumA[0][0]/NUM_PPL
	avgA[0][1] = sumA[0][1]/NUM_PPL
	avgA[1][0] = sumA[1][0]/NUM_PPL
	avgA[1][1] = sumA[1][1]/NUM_PPL

	print sumA[0][0]/NUM_PPL
	print sumA[1][0]/NUM_PPL

	avgB[0][0] = sumB[0][0]/NUM_PPL
	avgB[0][1] = sumB[0][1]/NUM_PPL
	avgB[1][0] = sumB[1][0]/NUM_PPL
	avgB[1][1] = sumB[1][1]/NUM_PPL

	stdA[0][0] = np.std(pplA[:,0,0])
	stdA[0][1] = np.std(pplA[:,0,1])
	stdA[1][0] = np.std(pplA[:,1,0])
	stdA[1][1] = np.std(pplA[:,1,1])

	stdB[0][0] = np.std(pplB[:,0,0])
	stdB[0][1] = np.std(pplB[:,0,1])
	stdB[1][0] = np.std(pplB[:,1,0])
	stdB[1][1] = np.std(pplB[:,1,1])

	print avgA

	xlabel = "Task"
	ylabel = r"Cost Value"
	title = r"Average Cost Across Tasks"	
	maxY = 200.0
	fig = plotting(avgA,avgB,stdA,stdB,xlabel,ylabel,title,maxY,optimal)

	if saveFig:
		here = os.path.dirname(os.path.realpath(__file__))
		subdir = "/data/experimental/"
		datapath = here + subdir
		fig.savefig(datapath+"taskCost.pdf", bbox_inches="tight")
		print "Saved cost figure."


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

def plot_weights(saveFig=False):
	weightData = data_io.parse_exp_data("weights")

	f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
	#f.set_title('Task 1: Table')
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

	task = 2
	trial = 2

	if task == 1:
		f.suptitle("Task 1: Correct Distance to Table",fontsize=20)
	else:
		f.suptitle("Task 2: Correct Distance to Table + Cup Orientation",fontsize=20)
	Acount = 0
	Bcount = 0

	"""
	for method in weightData[1][task][trial]:
		data = weightData[1][task][trial][method]
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
	"""

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


if __name__ == '__main__':
	#plot_taskEffort(saveFig=True)
	#plot_taskEffortTime(saveFig=True)
	#plot_taskCost(saveFig=True)
	plot_weights(saveFig=True)
