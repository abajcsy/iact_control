import numpy as np
from numpy import linalg
from numpy import linspace
import matplotlib.pyplot as plt
import time
import math
import json

from sympy import symbols
from sympy import lambdify

import trajoptpy
import or_trajopt
import openravepy
from openravepy import *

import openrave_utils
from openrave_utils import *

import logging
import pid
import copy

import grav_comp
from grav_comp import *


MAX_ITER = 40		#2 or 40

LAPTOP_1_xyz = [-1.3858/2.0 - 0.1, -0.3, 0.0]
LAPTOP_1_xy = np.array(LAPTOP_1_xyz[0:2])
LAPTOP_2_xyz = [-1.3858/2.0 + 0.35, 0.3, 0.0]
LAPTOP_2_xy = np.array(LAPTOP_2_xyz[0:2])
HUMAN_xyz = [-1.3858/2.0 - 0.1, -0.5, 0.0]
HUMAN_xy = np.array(HUMAN_xyz[0:2])



class Planner(object):


	def __init__(self):

		# ---- DEFORMATION Initialization ---- #

		self.alpha = -0.01
		self.n = 5
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

		# ---- important internal variables ---- #

		#these variables are fixed
		self.num_waypts_plan = 6

		#these variables change at each call
		self.weights = None
		self.waypts = None
		self.start_time = None
		self.final_time = None
		self.step_time = None

		# ---- OpenRAVE Initialization ---- #
		
		# initialize robot and empty environment
		model_filename = 'jaco_dynamics'
		self.env, self.robot = initialize(model_filename)

		# insert any objects you want into environment
		self.bodies = []
	
		# plot the table and table mount
		plotTable(self.env)
		plotTableMount(self.env,self.bodies)
		plotCabinet(self.env)
		#plotSphere(self.env,self.bodies,LAPTOP_1_xyz,0.3)
		#plotSphere(self.env,self.bodies,LAPTOP_2_xyz,0.3)
		#plotSphere(self.env,self.bodies,HUMAN_xyz,0.4)


	
	# -- Distance to Table -- #

	def table_features(self, waypt):

		if len(waypt) < 10:
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]), 1)
			waypt[2] += math.pi
		self.robot.SetDOFValues(waypt)
		EE_link = self.robot.GetLinks()[10]
		EE_coord_z = EE_link.GetTransform()[2][3]
		return EE_coord_z
	
	def table_cost(self, waypt):

		feature = self.table_features(waypt)
		return feature*self.weights


	# -- Face the Human -- #

	def mirror_features(self, waypt):

		if len(waypt) < 10:
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]), 1)
			waypt[2] += math.pi
		self.robot.SetDOFValues(waypt)
		EE_link = self.robot.GetLinks()[7]
		direction_y = sum(abs(EE_link.GetTransform()[1:3,0:3].dot([0.0,1.0,0.0])))
		direction_z = sum(abs(EE_link.GetTransform()[0:2,0:3].dot([1.0,0.0,0.0])))
		return direction_y + direction_z

	def mirror_cost(self, waypt):

		feature = self.mirror_features(waypt)
		return feature*self.weights


	# -- Distance to Laptop -- #

	def laptop_features(self, waypt, prev_waypt):

		feature = 0.0
		NUM_STEPS = 4
		for step in range(NUM_STEPS):
			inter_waypt = prev_waypt + (1.0 + step)/(NUM_STEPS)*(waypt - prev_waypt)
			feature += self.laptop_dist(inter_waypt)
		return feature

	def laptop_dist(self, waypt):

		if len(waypt) < 10:
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]), 1)
			waypt[2] += math.pi
		self.robot.SetDOFValues(waypt)
		EE_link = self.robot.GetLinks()[10]
		EE_coord_xy = EE_link.GetTransform()[0:2,3]
		dist_1 = np.linalg.norm(EE_coord_xy - LAPTOP_1_xy) - 0.3
		dist_2 = np.linalg.norm(EE_coord_xy - LAPTOP_2_xy) - 0.3
		dist = min([dist_1, dist_2])
		if dist > 0:
			return 0
		return -dist

	def laptop_cost(self, waypt):

		prev_waypt = waypt[0:7]
		curr_waypt = waypt[7:14]
		feature = self.laptop_features(curr_waypt,prev_waypt)
		return feature*self.weights*np.linalg.norm(curr_waypt - prev_waypt)


	# -- Distance to Human -- #

	def human_features(self, waypt, prev_waypt):

		feature = 0.0
		NUM_STEPS = 4
		for step in range(NUM_STEPS):
			inter_waypt = prev_waypt + (1.0 + step)/(NUM_STEPS)*(waypt - prev_waypt)
			feature += self.human_dist(inter_waypt)
		return feature

	def human_dist(self, waypt):

		if len(waypt) < 10:
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]), 1)
			waypt[2] += math.pi
		self.robot.SetDOFValues(waypt)
		EE_link = self.robot.GetLinks()[10]
		EE_coord_xy = EE_link.GetTransform()[0:2,3]
		dist = np.linalg.norm(EE_coord_xy - HUMAN_xy) - 0.4
		if dist > 0:
			return 0
		return -dist

	def human_cost(self, waypt):

		prev_waypt = waypt[0:7]
		curr_waypt = waypt[7:14]
		feature = self.human_features(curr_waypt,prev_waypt)
		return feature*self.weights*np.linalg.norm(curr_waypt - prev_waypt)


	# ---- Table Constraint --- #

	def table_constraint(self, waypt):

		if len(waypt) < 10:
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]), 1)
			waypt[2] += math.pi
		self.robot.SetDOFValues(waypt)
		EE_link = self.robot.GetLinks()[10]
		EE_coord_z = EE_link.GetTransform()[2][3]
		if EE_coord_z > 0:
			EE_coord_z = 0
		return -EE_coord_z




	# ---- here's trajOpt --- #
		
	def trajOpt(self, start, goal, start_time, final_time):
		"""
		Computes a plan from start to goal using trajectory optimizer.
		Reference: http://joschu.net/docs/trajopt-paper.pdf
		---
		input is start and goal pos, updates the waypts_plan
		"""
		if len(start) < 10:
			aug_start = np.append(start.reshape(7), np.array([0,0,0]), 1)
		self.robot.SetDOFValues(aug_start)


		init_waypts = np.zeros((self.num_waypts_plan,7))
		init_waypts[0,:] = start
		step_time = (final_time - start_time)/(self.num_waypts_plan - 1.0)
		for count in range(1, self.num_waypts_plan):
			curr_time = start_time + count*step_time
			if self.waypts == None:				
				init_waypts[count,:] = start + (curr_time - start_time)/(final_time - start_time)*(goal - start)
			else:
				init_waypts[count,:] = self.interpolate(curr_time).reshape((1,7))

		
		request = {
			"basic_info": {
				"n_steps": self.num_waypts_plan,
				"manip" : "j2s7s300",
				"max_iter": MAX_ITER
			},
			"costs": [
			{
				"type": "joint_vel",
				"params": {"coeffs": [1.0]}
			}
			],
			"constraints": [
			{
				"type": "joint",
				"params": {"vals": goal.tolist()}
#				"type" : "pose", 
#				"params" : {"xyz" : [-0.41992156, 0.52793478, 0.57166576], 
#							"wxyz" : [1,0,0,0],
#							"link": "j2s7s300_link_7",
#							"rot_coeffs" : [0,0,0],
#							"pos_coeffs" : [10,10,10]
#							}	 
			}			  
			],
			"init_info": {
                "type": "given_traj",
                "data": init_waypts.tolist()
			}
		}

		s = json.dumps(request)
		prob = trajoptpy.ConstructProblem(s, self.env)

		for t in range(1,self.num_waypts_plan):
			prob.AddCost(self.table_cost, [(t,j) for j in range(7)], "table%i"%t)
			#prob.AddCost(self.mirror_cost, [(t,j) for j in range(7)], "mirror%i"%t)
			#prob.AddCost(self.laptop_cost, [(t-1,j) for j in range(7)]+[(t,j) for j in range(7)], "laptop%i"%t)
			#prob.AddCost(self.human_cost, [(t-1,j) for j in range(7)]+[(t,j) for j in range(7)], "human%i"%t)
	
		for t in range(1,self.num_waypts_plan - 1):
			prob.AddConstraint(self.table_constraint, [(t,j) for j in range(7)], "INEQ", "up%i"%t)
		
		result = trajoptpy.OptimizeProblem(prob)
		self.waypts = result.GetTraj()



	# ---- update rule for weights ---- #

	def update(self, waypts_deform, waypts_prev):

		Phi_p = 0.0
		Phi = 0.0
		for count in range(1,waypts_deform.shape[0]):
			Phi_p += self.table_features(waypts_deform[count,:])
			Phi += self.table_features(waypts_prev[count,:])
			#Phi_p += self.laptop_features(waypts_deform[count,:], waypts_deform[count-1,:])
			#Phi += self.laptop_features(waypts_prev[count,:], waypts_prev[count-1,:])

		self.weights += -1.0*(Phi_p - Phi)
		return self.weights


	# ---- deform the desired trajectory ---- #	

	def deform(self, force, curr_time):

		deltaT = curr_time - self.start_time
		deform_waypt_idx = int(deltaT/self.step_time) + 1
		if (deform_waypt_idx + self.n) > self.waypts.shape[0]:
			return (None, None)
		waypts_prev = copy.deepcopy(self.waypts)
		waypts_deform = copy.deepcopy(self.waypts)
		gamma = np.zeros((self.n,7))
		for joint in range(7):
			gamma[:,joint] = self.alpha*np.dot(self.H, force[joint])
		waypts_deform[deform_waypt_idx : self.n + deform_waypt_idx, :] += gamma
		self.waypts = waypts_deform
		return (waypts_deform, waypts_prev)



	# ---- obtaining the desired position ---- #

	def replan(self, start, goal, start_time, final_time, weights):

		self.weights = weights
		self.trajOpt(start, goal, start_time, final_time)
		self.start_time = start_time
		self.final_time = final_time
		self.step_time = (self.final_time - self.start_time)/(self.num_waypts_plan - 1.0)


	def updateStart(self, start, elapsed_time):

		self.start_time += elapsed_time
		self.final_time += elapsed_time
		delta = start - self.waypts[0,:]
		if np.linalg.norm(delta) < 1e-3:
			return
		for count in range(self.num_waypts_plan):
			curr = self.waypts[count,:]
			self.waypts[count,:] = curr + (self.num_waypts_plan - 1.0 - count)/(self.num_waypts_plan - 1.0)*delta


	def upsample(self, step_time):

		num_waypts = int(math.ceil((self.final_time - self.start_time)/step_time)+0.001) + 1
		waypts = np.zeros((num_waypts,7))
		waypts[0,:] = self.waypts[0,:]
		for count in range(1, num_waypts):
			curr_time = self.start_time + count*step_time
			waypts[count,:] = self.interpolate(curr_time).reshape((1,7))
		self.step_time = step_time
		self.waypts = waypts
			

	def interpolate(self, curr_time):

		if curr_time >= self.final_time:
			target_pos = self.waypts[-1,:]
		else:
			deltaT = curr_time - self.start_time
			curr_waypt_idx = int(deltaT/self.step_time)
			prev = self.waypts[curr_waypt_idx]
			next = self.waypts[curr_waypt_idx + 1]
			deltaT_prev = curr_waypt_idx * self.step_time
			target_pos = (next - prev)*((deltaT - deltaT_prev)/self.step_time) + prev
		target_pos = np.array(target_pos).reshape((7,1))
		return target_pos



	# ---- close planner ---- #

	def kill_planner(self):
		"""
		Destroys openrave thread and environment for clean shutdown
		"""
		self.env.Destroy()
		RaveDestroy()

	

if __name__ == '__main__':

	time.sleep(50)

