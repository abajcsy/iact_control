import numpy as np
from numpy import linalg
from numpy import linspace
import matplotlib.pyplot as plt
import time
import math

from sympy import symbols
from sympy import lambdify

import trajoptpy
import or_trajopt
import openravepy
from openravepy import *

import interactpy
from interactpy import *
from util import *

import logging
import pid
import copy
import json

import sim_robot

logging.getLogger('prpy.planning.base').addHandler(logging.NullHandler())



class Planner(object):
	"""
	This class plans a trajectory from start to goal 
	with TrajOpt. 
	"""



	def __init__(self):

		# ---- important internal variables ---- #

		self.start_time = None
		self.final_time = None
		self.curr_waypt_idx = None

		self.waypts_plan = None
		self.num_waypts_plan = None
		self.step_time_plan = None

		self.step_time = None
		self.num_waypts = None
		self.waypts = None
		self.waypts_time = None

		# ---- OpenRAVE Initialization ---- #
		
		# initialize openrave and compute waypts
		model_filename = 'jaco_dynamics'
		# model_filename = 'jaco'
		self.env, self.robot = interact.initialize_empty(model_filename, empty=True)

		# insert any objects you want into environment
		self.bodies = []
		coords = [0.0, 0.4, 0.8]
		self.plotPoint(coords, 0.1)
		#coords = [-0.2, 0.2, 0.6]
		#self.plotPoint(coords, 0.1)

		viewer = self.env.GetViewer()
		viewer.SetSize(1000,1000)
		viewer.SetCamera([
			[0.,  0., -1., 1.],
			[1.,  0.,  0., 0.],
			[0., -1.,  1., 0.],
			[0.,  0.,  0., 1.]])
		viewer.SetBkgndColor([0.8,0.8,0.8])

		# ---- DEFORMATION Initialization ---- #

		self.alpha = 0.0#-0.005
		self.n = 5 # number of waypoints that will be deformed
		self.A = np.zeros((self.n+2, self.n)) 
		np.fill_diagonal(self.A, 1)
		for i in range(self.n):
			self.A[i+1][i] = -2
			self.A[i+2][i] = 1
		self.R = np.dot(self.A.T, self.A)
		Rinv = np.linalg.inv(self.R)
		Uh = np.zeros((self.n, 1))
		Uh[0] = 1
		self.H = np.dot(Rinv,Uh)*(np.sqrt(self.n)/np.linalg.norm(np.dot(Rinv,Uh)))





	# ---- let's replan a new trajectory ---- #


	def replan(self, start, goal, weights, start_time, final_time, step_time):

		self.start_time = start_time
		self.final_time = final_time
		self.curr_waypt_idx = 0
		self.trajOpt(start, goal, weights)
		self.step_time_plan = (self.final_time - self.start_time)/(self.num_waypts_plan - 1)
		self.upsample(step_time)



	def upsample(self, step_time):

		num_waypts = int(math.ceil((self.final_time - self.start_time)/step_time)) + 1
		waypts = np.zeros((num_waypts,7))
		waypts_time = [None]*num_waypts

		t = self.start_time
		for i in range(num_waypts):

			if t == self.final_time:
				
				waypts_time[i] = self.final_time
				waypts[i,:] = self.waypts_plan[self.num_waypts_plan - 1,:]

			else:

				deltaT = t - self.start_time
				lower_waypt_idx = int(deltaT/self.step_time_plan)
				lower_waypt = self.waypts_plan[lower_waypt_idx,:]
				higher_waypt = self.waypts_plan[lower_waypt_idx + 1,:]

				waypts_time[i] = t
				waypts[i,:] = lower_waypt+((t-lower_waypt_idx*self.step_time_plan)/self.step_time_plan)*(higher_waypt-lower_waypt)

			t += step_time
			if t > self.final_time:
				t = self.final_time

		self.step_time = step_time
		self.num_waypts = num_waypts
		self.waypts = waypts
		self.waypts_time = waypts_time



	def trajOpt(self, start, goal, weights):
		"""
		Computes a plan from start to goal taking T total time.

		"""

		if len(start) < 10:
			padding = np.array([0,0,0])
			aug_start = np.append(start.reshape(7), padding, 1)
		self.robot.SetDOFValues(aug_start)

		n_waypoints = 10
		w_length = weights[0]
		w_collision = weights[1]

		if self.waypts_plan == None:
			init_waypts = np.zeros((n_waypoints,7))
			for count in range(n_waypoints):
				init_waypts[count,:] = start + count/(n_waypoints - 1.0)*(goal - start)
		else:
			init_waypts = self.waypts_plan 
		self.num_waypts_plan = n_waypoints

		request = {
			"basic_info": {
				"n_steps": n_waypoints,
				"manip" : "j2s7s300"
			},
			"costs": [
			{
				"type": "joint_vel",
				"params": {"coeffs": [w_length]}
			},
			{
				"type": "collision",
				"params": {
				"coeffs": [w_collision],
				"dist_pen": [0.5]
				},
			}
			],
			"constraints": [
			{
				"type": "joint",
				"params": {"vals": goal.tolist()}
			}
			],
			"init_info": {
                "type": "given_traj",
                "data": init_waypts.tolist()
			}
		}

		s = json.dumps(request)
		prob = trajoptpy.ConstructProblem(s, self.env) #maybe we could get useful stuff from prob? some object?
		result = trajoptpy.OptimizeProblem(prob) #can use result to check our own rolled functions?
		self.waypts_plan = result.GetTraj()




	# ---- let's find the target position ---- #


	def interpolate(self, curr_time):
		"""
		Gets the next desired position along trajectory
		by interpolating between waypoints given the current t.
		"""
		target_pos = self.waypts[0]
		if self.num_waypts >= 2:
			for i in range(self.num_waypts-1):
				if curr_time > self.waypts_time[i] and curr_time < self.waypts_time[i+1]:
					prev = self.waypts[i]
					next = self.waypts[i+1]
					ti = self.waypts_time[i]
					ti1 = self.waypts_time[i+1]
					deltaT = ti1 - ti
					theta = (next - prev)*((curr_time-ti)/deltaT) + prev
					target_pos = theta
					self.curr_waypt_idx = i+1
					break
				elif curr_time == self.waypts_time[i]:
					target_pos = self.waypts[i]		
					self.curr_waypt_idx = i
					break
		else:
			print "ONLY ONE WAYPT, CAN'T INTERPOLATE."
	
		if curr_time > self.final_time:
			print "TIME IS UP. GOING TO FINAL WAYPOINT."
			target_pos = self.waypts[self.num_waypts-1]	
			self.curr_waypt_idx = self.num_waypts-1

		target_pos = np.array(target_pos).reshape((7,1))
		return target_pos




	# ---- let's deform the trajectory ---- #


	def deform(self, u_h):

		if (self.curr_waypt_idx + self.n) >= self.num_waypts:
			return

		gamma = np.zeros((self.n,7))
		for joint in range(7):
			gamma[:,joint] = self.alpha*np.dot(self.H, u_h[joint])

		gamma_prev = self.waypts[self.curr_waypt_idx : self.n + self.curr_waypt_idx, :]
		self.waypts[self.curr_waypt_idx : self.n + self.curr_waypt_idx, :] = gamma_prev + gamma












	# ------- Plotting Utils ------- #

	def plotTraj(self):
		"""
		Plots the best trajectory found or planned

		TODO: traj_pts plotting is deprecated!
		"""
		for body in self.bodies:
			self.env.Remove(body)

		self.bodies += plotWaypoints(self.env, self.robot, self.waypts)

	def plotPoint(self, coords, size=0.1):
		"""
		Plots a single cube point in OpenRAVE at coords(x,y,z) location
		"""
		with self.env:
			color = np.array([0, 1, 0])

			body = RaveCreateKinBody(self.env, '')
			body.InitFromBoxes(np.array([[coords[0], coords[1], coords[2],
						  size, size, size]]))
			body.SetName(str(len(self.bodies)))
			self.env.Add(body, True)
			body.GetLinks()[0].GetGeometries()[0].SetDiffuseColor(color)
			self.bodies.append(body)

	def plot_cartesian_waypts(self, cartesian):
		"""
		Plots cartesian waypoints in OpenRAVE.
		"""
		for i in range(self.num_waypts):
			self.plotPoint(cartesian[i])

	def get_cartesian_waypts(self):
		"""
		Returns list of waypoints along trajectory in task-space
		- Return type: list of length 3 numpy arrays
		"""
		cartesian = []
		for i in range(self.num_waypts):
			waypoint = self.waypts.GetWaypoint(i)
			dof = np.append(waypoint, np.array([1,1,1]))
			tf = transformToCartesian(dofToTransform(self.robot, dof))
			cartesian.append(tf)
		return np.array(cartesian)

	def execute_path_sim(self):
		"""
		Executes in simulation the planned trajectory
		"""
		self.robot.ExecutePath(self.waypts)

	def update_curr_pos(self, curr_pos):
		"""
		Updates DOF values in OpenRAVE simulation based on curr_pos.
		----
		curr_pos 	7x1 vector of current joint angles (degrees)
		----
		"""
		pos = np.array([curr_pos[0][0],curr_pos[1][0],curr_pos[2][0],curr_pos[3][0],curr_pos[4][0],curr_pos[5][0],curr_pos[6][0],0,0,0])
		self.curr_pos = pos

		self.robot.SetDOFValues(self.curr_pos)

if __name__ == '__main__':

	trajplanner = Planner()

	candlestick = np.array([180.0]*7)	
	home_pos = np.array([103.366,197.13,180.070,43.4309,265.11,257.271,287.9276])

	s = np.array(home_pos)*(math.pi/180.0)
	g = np.array(candlestick)*(math.pi/180.0)
	T = 8.0
	features = None
	weights = None

	startT = time.time()
	trajplanner.plan(s,g,features, weights, T)
	endT = time.time()
	print "Replanning took: " + str(endT - startT) + " seconds"

	trajplanner.execute_path_sim()
	time.sleep(20)



