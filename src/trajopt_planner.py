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

#import interactpy
#from interactpy import *

import openrave_utils
from openrave_utils import *

import logging
import pid
import copy

# possible range of feature values for each feature
CUP_RANGE = 17.12161451
TABLE_RANGE = 10.28238132
LAPTOP_RANGE = 9.99367477

HUMAN_TASK = 0
COFFEE_TASK = 1
TABLE_TASK = 2
LAPTOP_TASK = 3

OBS_CENTER = [-1.3858/2.0 - 0.1, -0.1, 0.0]
HUMAN_CENTER = [0.0, 0.2, 0.0]

# feature learning methods
ALL = 0 						# updates all features
MAX = 1							# updates only feature that changed the most
LIKELY = 2						# updates the most likely feature 

class Planner(object):
	"""
	This class plans a trajectory from start to goal 
	with TrajOpt. 
	"""

	def __init__(self, task, demo, featMethod):

		# ---- important internal variables ---- #

		self.task = task
		self.demo = demo
		if self.demo:
			self.MAX_ITER = 40
		else:
			self.MAX_ITER = 40

		self.featMethod = featMethod	# can be ALL, MAX, or LIKELY

		self.start_time = None
		self.final_time = None
		self.curr_waypt_idx = None

		# these variables are for trajopt
		self.waypts_plan = None
		self.num_waypts_plan = None
		self.step_time_plan = None

		# these variables are for the upsampled trajectory
		self.waypts = None
		self.num_waypts = None
		self.step_time = None
		self.waypts_time = None

		self.weights = [0.0,0.0,0.0]
		self.waypts_prev = None

		# ---- Plotting weights & features over time ---- #
		self.weight_update = None
		self.update_time = None

		self.feature_update = None
		self.prev_features = None
		self.curr_features = None
		self.update_time2 = None

		# ---- OpenRAVE Initialization ---- #
		
		# initialize robot and empty environment
		model_filename = 'jaco_dynamics'
		self.env, self.robot = initialize(model_filename)

		# insert any objects you want into environment
		self.bodies = []
	
		# plot the table and table mount
		plotTable(self.env)
		plotTableMount(self.env,self.bodies)
		plotLaptop(self.env,self.bodies)
		plotCabinet(self.env)
		#plotSphere(self.env,self.bodies,OBS_CENTER,0.4)
		#plotSphere(self.env,self.bodies,HUMAN_CENTER,0.4)
	
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

	# ---- utilities/getter functions ---- #

	def get_waypts_plan(self):
		"""
		Returns reference to waypts_plan (used by trajopt)
		Used mostly for recording experimental data by pid_trajopt.py
		"""
		return self.waypts_plan


	# ---- custom feature and cost functions ---- #

	def featurize(self, waypts):
		"""
		Computes the user-defined features for a given trajectory.
		---
		input trajectory, output list of feature values
		"""
		features = [None,None,None,None]
		features[0] = self.velocity_features(waypts)
		features[1] = [0.0]*(len(waypts)-1)
		features[2] = [0.0]*(len(waypts)-1)
		features[3] = [0.0]*(len(waypts)-1)
		for index in range(0,len(waypts)-1):
			features[1][index] = self.coffee_features(waypts[index+1])
			features[2][index] = self.table_features(waypts[index+1])
			features[3][index] = self.laptop_features(waypts[index+1],waypts[index])
			#elif self.task == HUMAN_TASK:
			#	features[1][index] = self.human_features(waypts[index+1],waypts[index])
		return features

	# -- Velocity -- #

	def velocity_features(self, waypts):
		"""
		Computes total velocity cost over waypoints, confirmed to match trajopt.
		---
		input trajectory, output scalar feature
		"""
		vel = 0.0
		for i in range(1,len(waypts)):
			curr = waypts[i]
			prev = waypts[i-1]
			vel += np.linalg.norm(curr - prev)**2
		return vel
	
	def velocity_cost(self, waypts):
		"""
		Computes the total velocity cost.
		---
		input trajectory, output scalar cost
		"""
		#mywaypts = np.reshape(waypts,(7,self.num_waypts_plan)).T
		return self.velocity_features(mywaypts)

	# -- Distance to Table -- #

	def table_features(self, waypt):
		"""
		Computes the total cost over waypoints based on 
		z-axis distance to table
		---
		input trajectory, output scalar feature
		"""
		if len(waypt) < 10:
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]), 1)
			waypt[2] += math.pi
		self.robot.SetDOFValues(waypt)
		coords = robotToCartesian(self.robot)
		EEcoord_z = coords[6][2]
		return EEcoord_z
	
	def table_cost(self, waypt):
		"""
		Computes the total distance to table cost.
		---
		input trajectory, output scalar cost
		"""
		feature = self.table_features(waypt)
		return feature*self.weights[1]

	# -- Coffee (or z-orientation of end-effector) -- #

	def coffee_features(self, waypt):
		"""
		Computes the distance to table cost for waypoint
		by checking if the EE is oriented vertically.
		Note: [0,0,1] in the first *column* corresponds to the cup upright
		---
		input trajectory, output scalar cost
		"""
		if len(waypt) < 10:
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]), 1)
			waypt[2] += math.pi
		self.robot.SetDOFValues(waypt)
		EE_link = self.robot.GetLinks()[7]
		return sum(abs(EE_link.GetTransform()[:2,:3].dot([1,0,0])))

		# ---------------------
		"""
		# plot base tf
		base_tf = self.robot.GetLinks()[0].GetTransform()
		#plotMug(self.env, self.bodies, base_tf)

		angle = np.pi/2
		rot_x = np.array([[1,0,0,0],[0,np.cos(angle),-np.sin(angle),0],[0,np.sin(angle),np.cos(angle),0],[0,0,0,1]]) 
		rot_y = np.array([[np.cos(angle),0,np.sin(angle),0],[0,1,0,0],[-np.sin(angle),0,np.cos(angle),0],[0,0,0,1]]) 
		rot_z = np.array([[np.cos(angle),-np.sin(angle),0,0],[np.sin(angle),0,np.cos(angle),0],[0,0,1,0],[0,0,0,1]]) 
		trans = np.array([[0,0,0,0.05],[0,0,0,0.05],[0,0,0,0.05],[0,0,0,1]]) 

		# to rotate cup upright --> angle = pi/2, rot_x
		# to rotate cup down --> angle = -pi/2, rot_x
		rot_base_tf = np.dot(base_tf, rot_x)
		base_mug = plotMug(self.env, self.bodies, rot_base_tf, color=[0,0,1])

		# plot EE tf
		ee_tf = self.robot.GetLinks()[6].GetTransform()
		plotMug(self.env, self.bodies, ee_tf, color=[0,1,0])
		rot_ee_tf = np.dot(ee_tf, rot_z)
		plotMug(self.env, self.bodies, rot_ee_tf)

		base_rot = base_mug.GetTransform()[:3,:3] 
		base_axis_up = openravepy.axisAngleFromRotationMatrix(base_rot)
		wrist_rot = self.robot.GetLinks()[6].GetTransform()[:3,:3]
		wrist_axis = openravepy.axisAngleFromRotationMatrix(wrist_rot)

		#base_axis = np.array(self.robot.GetJoints()[0].GetAxis())
		#EE_axis = np.array(self.robot.GetJoints()[2].GetAxis())
		#print "base_axis: " + str(base_axis)
		#print "EE_axis: " + str(EE_axis)

		#ideal_angle = np.array([0, np.pi, 0])
		#wrist_angle = openravepy.axisAngleFromRotationMatrix(self.robot.GetLinks()[7].GetTransform()[:3,:3])

		#tf = EE_link.GetTransform()[:3,:3]
		#r_x = abs(tf[1][1])+abs(tf[2][1]) #+abs(tf[0][1]) 
		#r_y = abs(tf[0][2])+abs(tf[2][2]) #+abs(tf[1][2])
		#r_z = abs(tf[0][0])+abs(tf[1][0]) #+abs(tf[2][0])
		#coffee_feature = r_z

		coffee_feature = abs(np.dot(base_axis_up, wrist_axis))
		print "coffee feature: " + str(coffee_feature)

		return coffee_feature
		"""
		
	def coffee_cost(self, waypt):
		"""
		Computes the total coffee (EE orientation) cost.
		---
		input trajectory, output scalar cost
		"""
		feature = self.coffee_features(waypt)
		return feature*self.weights[0]

	# -- Distance to Laptop -- #

	def laptop_features(self, waypt, prev_waypt):
		"""
		Computes laptop cost over waypoints, interpolating and
		sampling between each pair to check for intermediate collisions
		---
		input trajectory, output scalar feature
		"""
		feature = 0.0
		NUM_STEPS = 4
		for step in range(NUM_STEPS):
			inter_waypt = prev_waypt + (1.0 + step)/(NUM_STEPS)*(waypt - prev_waypt)
			feature += self.laptop_dist(inter_waypt)
		return feature

	def laptop_dist(self, waypt):
		"""
		Computes distance from end-effector to laptop in xy coords
		input trajectory, output scalar distance where 
			0: EE is at more than 0.4 meters away from laptop
			+: EE is closer than 0.4 meters to laptop
		"""
		if len(waypt) < 10:
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]), 1)
			waypt[2] += math.pi
		self.robot.SetDOFValues(waypt)
		coords = robotToCartesian(self.robot)
		EE_coord_xy = coords[6][0:2]
		laptop_xy = np.array(OBS_CENTER[0:2])
		dist = np.linalg.norm(EE_coord_xy - laptop_xy) - 0.4
		if dist > 0:
			return 0
		return -dist

	def laptop_cost(self, waypt):
		"""
		Computes the total distance to laptop cost
		---
		input trajectory, output scalar cost
		"""
		prev_waypt = waypt[0:7]
		curr_waypt = waypt[7:14]
		feature = self.laptop_features(curr_waypt,prev_waypt)
		return feature*self.weights[2]*np.linalg.norm(curr_waypt - prev_waypt)

	# -- Distance to Human -- #

	def human_features(self, waypt, prev_waypt):
		"""
		Computes laptop cost over waypoints, interpolating and
		sampling between each pair to check for intermediate collisions
		---
		input trajectory, output scalar feature
		"""
		feature = 0.0
		NUM_STEPS = 4
		for step in range(NUM_STEPS):
			inter_waypt = prev_waypt + (1.0 + step)/(NUM_STEPS)*(waypt - prev_waypt)
			feature += self.human_dist(inter_waypt)
		return feature

	def human_dist(self, waypt):
		"""
		Computes distance from end-effector to laptop in xy coords
		input trajectory, output scalar distance where 
			0: EE is at more than 0.4 meters away from laptop
			+: EE is closer than 0.4 meters to laptop
		"""
		if len(waypt) < 10:
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]), 1)
			waypt[2] += math.pi
		self.robot.SetDOFValues(waypt)
		coords = robotToCartesian(self.robot)
		EE_coord_xy = coords[6][0:2]
		human_xy = np.array(HUMAN_CENTER[0:2])
		dist = np.linalg.norm(EE_coord_xy - human_xy) - 0.4
		if dist > 0:
			return 0
		return -dist

	def human_cost(self, waypt):
		"""
		Computes the total distance to laptop cost
		---
		input trajectory, output scalar cost
		"""
		prev_waypt = waypt[0:7]
		curr_waypt = waypt[7:14]
		feature = self.human_features(curr_waypt,prev_waypt)
		return feature*self.weights*np.linalg.norm(curr_waypt - prev_waypt)


	# ---- custom constraints --- #

	def table_constraint(self, waypt):
		"""
		Constrains z-axis of robot's end-effector to always be 
		above the table.
		"""
		if len(waypt) < 10:
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]), 1)
			waypt[2] += math.pi
		self.robot.SetDOFValues(waypt)
		EE_link = self.robot.GetLinks()[10]
		EE_coord_z = EE_link.GetTransform()[2][3]
		if EE_coord_z > 0:
			EE_coord_z = 0
		return -EE_coord_z

	def coffee_constraint(self, waypt):
		"""
		Constrains orientation of robot's end-effector to be 
		holding coffee mug upright.
		"""
		if len(waypt) < 10:
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]), 1)
			waypt[2] += math.pi
		self.robot.SetDOFValues(waypt)
		EE_link = self.robot.GetLinks()[7]
		return EE_link.GetTransform()[:2,:3].dot([1,0,0])

		"""
		tf = EE_link.GetTransform()[:3,:3]

		r_x = abs(tf[1][1])+abs(tf[2][1]) #+abs(tf[0][1]) 
		r_y = abs(tf[0][2])+abs(tf[2][2]) #+abs(tf[1][2])
		r_z = abs(tf[0][0])+abs(tf[1][0]) #+abs(tf[2][0])

		coffee_feature = r_z

		return coffee_feature
		"""
		
	def coffee_constraint_derivative(self, waypt):
		"""
		Analytic derivative for coffee constraint.
		"""
		if len(waypt) < 10:
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]), 1)
			waypt[2] += math.pi
		self.robot.SetDOFValues(waypt)
		world_dir = self.robot.GetLinks()[7].GetTransform()[:3,:3].dot([1,0,0])
		return np.array([np.cross(self.robot.GetJoints()[i].GetAxis(), world_dir)[:2] for i in range(7)]).T.copy()


	# ---- here's trajOpt --- #
		
	def trajOptPose(self, start, goal, goal_pose):
		"""
		Computes a plan from start to goal using trajectory optimizer.
		Goal is a pose, not a configuration!
		Reference: http://joschu.net/docs/trajopt-paper.pdf
		---
		input:
			start and goal pos, and a trajectory to seed trajopt with
		return: 
			the waypts_plan trajectory 
		"""

		print "I'm in trajopt_PLANNER trajopt pose!"		

		# plot goal point
		#plotSphere(self.env, self.bodies, goal_pose, size=20)
	
		if len(start) < 10:
			aug_start = np.append(start.reshape(7), np.array([0,0,0]), 1)
		self.robot.SetDOFValues(aug_start)

		self.num_waypts_plan = 4

		xyz_target = goal_pose
		quat_target = [1,0,0,0] # wxyz
	
		init_joint_target =  goal 

		request = {
			"basic_info": {
				"n_steps": self.num_waypts_plan,
				"manip" : "j2s7s300",
				"start_fixed" : True,
				"max_iter" : self.MAX_ITER
			},
			"costs": [
			{
				"type": "joint_vel",
				"params": {"coeffs": [1.0]}
			}
			],
			"constraints": [
			{
				"type": "pose",
				"params" : {"xyz" : xyz_target, 
						    "wxyz" : quat_target, 
						    "link": "j2s7s300_link_7",
							"rot_coeffs" : [0,0,0],
							"pos_coeffs" : [100,100,100],
						    }
			}
			],
			"init_info": {
                "type": "straight_line",
                "endpoint": init_joint_target.tolist()
			}
		}

		s = json.dumps(request)
		prob = trajoptpy.ConstructProblem(s, self.env)

		for t in range(1,self.num_waypts_plan):
			prob.AddCost(self.coffee_cost, [(t,j) for j in range(7)], "coffee%i"%t)
			prob.AddCost(self.table_cost, [(t,j) for j in range(7)], "table%i"%t)
			prob.AddCost(self.laptop_cost, [(t-1,j) for j in range(7)]+[(t,j) for j in range(7)], "laptop%i"%t)

		for t in range(1,self.num_waypts_plan - 1):
			prob.AddConstraint(self.table_constraint, [(t,j) for j in range(7)], "INEQ", "table%i"%t)
			#prob.AddConstraint(self.coffee_constraint, [(t,j) for j in range(7)], "INEQ", "coffee%i"%t)

		result = trajoptpy.OptimizeProblem(prob)
		self.waypts_plan = result.GetTraj()
		self.step_time_plan = (self.final_time - self.start_time)/(self.num_waypts_plan - 1)

		# plot resulting trajectory
		#plotTraj(self.env,self.robot,self.bodies,self.waypts_plan, size=10,color=[0, 0, 1])
		plotCupTraj(self.env,self.robot,self.bodies,self.waypts_plan,color=[0,1,0])		
		#time.sleep(10)

		print "I'm done with trajopt pose!"

		
		return self.waypts_plan


	def trajOpt(self, start, goal, traj_seed=None):
		"""
		Computes a plan from start to goal using trajectory optimizer.
		Reference: http://joschu.net/docs/trajopt-paper.pdf
		---
		input is start and goal pos, updates the waypts_plan
		"""

		print "I'm in normal trajOpt!"

		if len(start) < 10:
			aug_start = np.append(start.reshape(7), np.array([0,0,0]), 1)
		self.robot.SetDOFValues(aug_start)

		self.num_waypts_plan = 4

		# --- linear interpolation seed --- #
		if traj_seed is None:
			print "using straight line!"
			init_waypts = np.zeros((self.num_waypts_plan,7))
			for count in range(self.num_waypts_plan):
				init_waypts[count,:] = start + count/(self.num_waypts_plan - 1.0)*(goal - start)
		else:
			print "using traj seed!"
			# TODO THIS IS EXPERIMENTAL
			init_waypts = traj_seed#traj_seed.waypts_plan
			print init_waypts
		
		request = {
			"basic_info": {
				"n_steps": self.num_waypts_plan,
				"manip" : "j2s7s300",
				"max_iter" : self.MAX_ITER
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
			prob.AddCost(self.coffee_cost, [(t,j) for j in range(7)], "coffee%i"%t)
			prob.AddCost(self.table_cost, [(t,j) for j in range(7)], "table%i"%t)
			prob.AddCost(self.laptop_cost, [(t-1,j) for j in range(7)]+[(t,j) for j in range(7)], "laptop%i"%t)
			#elif self.task == HUMAN_TASK:
			#	prob.AddCost(self.human_cost, [(t-1,j) for j in range(7)]+[(t,j) for j in range(7)], "human%i"%t)
			#prob.AddErrorCost(self.laptop_cost, [(t-1,j) for j in range(7)]+[(t,j) for j in range(7)], "HINGE", "laptop%i"%t)

		for t in range(1,self.num_waypts_plan - 1):
			prob.AddConstraint(self.table_constraint, [(t,j) for j in range(7)], "INEQ", "up%i"%t)
			#prob.AddConstraint(self.laptop_cost, [(t-1,j) for j in range(7)]+[(t,j) for j in range(7)], "EQ", "up%i"%t)
			#prob.AddConstraint(self.coffee_constraint, self.coffee_constraint_derivative, [(t,j) for j in range(7)], "EQ", "up%i"%t)
			#prob.AddConstraint(self.coffee_constraint, [(t,j) for j in range(7)], "INEQ", "coffee%i"%t)

		result = trajoptpy.OptimizeProblem(prob)
		self.waypts_plan = result.GetTraj()
		self.step_time_plan = (self.final_time - self.start_time)/(self.num_waypts_plan - 1)

		#plotTraj(self.env,self.robot,self.bodies,self.waypts_plan, size=10,color=[0, 0, 1])
		plotCupTraj(self.env,self.robot,self.bodies,self.waypts_plan,color=[0,1,0])


	# ---- here's our algorithms for modifying the trajectory ---- #

	def computeDeltaQDes(self, start, goal):
		"""
		Computes the optimal change in configuration space (delta Q) for each feature.		
		Returns a dictionary indexed by feature and direction
		"""
		num_features = len(self.weights)
		# save current weights		
		w_curr = [self.weights[i] for i in range(num_features)]		
		deltaQdes = {}
		for f_idx in range(num_features):
			deltaQdes[f_idx] = {}
			for direction in [-1,1]:
				delta = 2.0
				delta_w = [0.0]*num_features
				delta_w[f_idx] = delta*self.weights[f_idx]
				# update weights based on ideal correction for given features and direction
				self.weights = w_curr + direction*delta_w
				deltaQdes[f_idx][direction] = traj - self.waypts_plan 
		# resets the current weights to original weights
		self.weights = w_curr
		return deltaQdes

	def learnWeights(self, u_h):
		"""
		Deforms the trajectory given human force, u_h, and
		updates features by computing difference between 
		features of new trajectory and old trajectory
		---
		input is human force and returns updated weights 
		"""
		(waypts_deform, waypts_prev) = self.deform(u_h)	
		if waypts_deform != None:
			new_features = self.featurize(waypts_deform)
			old_features = self.featurize(waypts_prev)
			Phi_p = np.array([new_features[0], sum(new_features[1]), sum(new_features[2]), sum(new_features[3])])
			Phi = np.array([old_features[0], sum(old_features[1]), sum(old_features[2]), sum(old_features[3])])
			
			self.prev_features = Phi_p
			self.curr_features = Phi

			# [update_gain_coffee, update_gain_table, update_gain_laptop] 
			update_gains = [2.0, 2.0, 10.0]

			# [max_weight_coffee, max_weight_table, max_weight_laptop] 
			max_weights = [1.0, 1.0, 10.0] 

			update = Phi_p - Phi
			print "Phi prev: " + str(Phi_p)
			print "Phi curr: " + str(Phi)
			print "Phi_p - Phi = " + str(update)
				
			if self.featMethod == ALL:
				# update all weights 
				curr_weight = [self.weights[0] - update_gains[0]*update[1], self.weights[1] - update_gains[1]*update[2], self.weights[2] - update_gains[2]*update[3]]	
	
			elif self.featMethod == MAX:
				change_in_features = [update[1]/CUP_RANGE, update[2]/TABLE_RANGE, update[3]/LAPTOP_RANGE]

				# get index of maximal change
				max_idx = np.argmax(np.fabs(change_in_features))
				
				# update only weight of feature with maximal change
				curr_weight = [self.weights[i] for i in range(len(self.weights))]
				curr_weight[max_idx] = curr_weight[max_idx] - update_gains[max_idx]*update[max_idx+1]

			print "curr_weight after = " + str(curr_weight)

			# clip values at max and min allowed weights
			for i in range(3):
				curr_weight[i] = np.clip(curr_weight[i], -max_weights[i], max_weights[i])

			print "here are the old weights:", self.weights
			print "here are the new weights:", curr_weight

			self.weights = curr_weight
			return self.weights

	def deform(self, u_h):
		"""
		Deforms the next n waypoints of the upsampled trajectory
		updates the upsampled trajectory, stores old trajectory
		---
		input is human force, returns deformed and old waypts
		"""
		deform_waypt_idx = self.curr_waypt_idx + 1
		if (deform_waypt_idx + self.n) > self.num_waypts:
			return (None, None)
		waypts_prev = copy.deepcopy(self.waypts)
		waypts_deform = copy.deepcopy(self.waypts)
		gamma = np.zeros((self.n,7))
		for joint in range(7):
			gamma[:,joint] = self.alpha*np.dot(self.H, u_h[joint])
		waypts_deform[deform_waypt_idx : self.n + deform_waypt_idx, :] += gamma
		#plotTraj(self.env, self.robot, self.bodies, self.waypts_plan, [1, 0, 0])
		return (waypts_deform, waypts_prev)
	

	# ---- replanning, upsampling, and interpolating ---- #

	def replan(self, start, goal, weights, start_time, final_time, step_time, seed=None):
		"""
		Replan the trajectory from start to goal given weights.
		---
		input trajectory parameters, update raw and upsampled trajectories
		"""
		if weights == None:
			return
		self.start_time = start_time
		self.final_time = final_time
		self.curr_waypt_idx = 0
		self.weights = weights
		
		# TODO THIS IS EXPERIMENTAL
		#place_pose = [-0.46513, 0.29041, 0.69497]
		#self.trajOptPose(start, goal, place_pose)
		self.trajOpt(start, goal, traj_seed=seed)
		
		#print "waypts_plan after trajopt: " + str(self.waypts_plan)
		self.upsample(step_time)
		#print "waypts_plan after upsampling: " + str(self.waypts_plan)
		#plotTraj(self.env,self.robot,self.bodies,self.waypts_plan, [0, 0, 1])

		return self.waypts_plan

	def upsample(self, step_time):
		"""
		Put waypoints along trajectory at step_time increments.
		---
		input desired time increment, update upsampled trajectory
		"""
		num_waypts = int(math.ceil((self.final_time - self.start_time)/step_time)) + 1
		waypts = np.zeros((num_waypts,7))
		waypts_time = [None]*num_waypts
		
		t = self.start_time
		for i in range(num_waypts):
			if t >= self.final_time:
				waypts_time[i] = self.final_time
				waypts[i,:] = self.waypts_plan[self.num_waypts_plan - 1]
			else:
				deltaT = t - self.start_time
				prev_idx = int(deltaT/self.step_time_plan)
				prev = self.waypts_plan[prev_idx]
				next = self.waypts_plan[prev_idx + 1]
				waypts_time[i] = t
				waypts[i,:] = prev+((t-prev_idx*self.step_time_plan)/self.step_time_plan)*(next-prev)
			t += step_time
		self.step_time = step_time
		self.num_waypts = num_waypts
		self.waypts = waypts
		self.waypts_time = waypts_time

	def downsample(self):
		"""
		Updates the trajopt trajectory from the upsampled trajectory.
		changes the trajopt waypoints between start and goal.
		"""
		for index in range(1,self.num_waypts_plan-1):
			t = self.start_time + index*self.step_time_plan
			target_pos = self.interpolate(t)
			self.waypts_plan[index,:] = target_pos.reshape((1,7))
			
	def interpolate(self, curr_time):
		"""
		Gets the next desired position along trajectory
		by interpolating between waypoints given the current t.
		"""
		if curr_time >= self.final_time:
			self.curr_waypt_idx = self.num_waypts - 1
			target_pos = self.waypts[self.curr_waypt_idx]
		else:
			deltaT = curr_time - self.start_time
			self.curr_waypt_idx = int(deltaT/self.step_time)
			prev = self.waypts[self.curr_waypt_idx]
			next = self.waypts[self.curr_waypt_idx + 1]
			ti = self.waypts_time[self.curr_waypt_idx]
			tf = self.waypts_time[self.curr_waypt_idx + 1]
			target_pos = (next - prev)*((curr_time-ti)/(tf - ti)) + prev		
		target_pos = np.array(target_pos).reshape((7,1))
		return target_pos

	def update_curr_pos(self, curr_pos):
		"""
		Updates DOF values in OpenRAVE simulation based on curr_pos.
		----
		curr_pos - 7x1 vector of current joint angles (degrees)
		"""
		pos = np.array([curr_pos[0][0],curr_pos[1][0],curr_pos[2][0]+math.pi,curr_pos[3][0],curr_pos[4][0],curr_pos[5][0],curr_pos[6][0],0,0,0])
		
		self.robot.SetDOFValues(pos)

	def plot_weight_update(self):
		"""
		Plots weight update over time.
		"""
		
		#plt.plot(self.update_time,self.weight_update.T[0],linewidth=4.0,label='Vel')
		plt.plot(self.update_time,self.weight_update.T[0],linewidth=4.0,label='Coffee')
		plt.plot(self.update_time,self.weight_update.T[1],linewidth=4.0,label='Table')
		plt.plot(self.update_time,self.weight_update.T[2],linewidth=4.0,label='Laptop')
		plt.legend()
		plt.title("Weight (for features) changes over time")
		plt.show()		

	def plot_feature_update(self):
		"""
		Plots feature change over time.
		"""
		
		#plt.plot(self.update_time,self.weight_update.T[0],linewidth=4.0,label='Vel')
		plt.plot(self.update_time2,self.feature_update.T[1],linewidth=4.0,label='Coffee')
		plt.plot(self.update_time2,self.feature_update.T[2],linewidth=4.0,label='Table')
		plt.plot(self.update_time2,self.feature_update.T[3],linewidth=4.0,label='Laptop')
		plt.legend()
		plt.title("Feature changes over time")
		plt.show()		

	def kill_planner(self):
		"""
		Destroys openrave thread and environment for clean shutdown
		"""
		self.env.Destroy()
		RaveDestroy() # destroy the runtime
	
if __name__ == '__main__':

	pick_basic = [104.2, 151.6, 183.8, 101.8, 224.2, 216.9, 310.8]
	place_lower = [210.8, 101.6, 192.0, 114.7, 222.2, 246.1, 322.0]

	pick_basic_EEtilt = [104.2, 151.6, 183.8, 101.8, 224.2, 216.9, 200.0] #200.0]
	place_lower_EEtilt = [210.8, 101.6, 192.0, 114.7, 222.2, 246.1, 400.0]

	#place_pose = [-0.58218719,  0.33018986,  0.10592141] # x, y, z for pick_lower_EEtilt

	# initialize start/goal based on task 
	pick = pick_basic_EEtilt
	place = place_lower 

	start = np.array(pick)*(math.pi/180.0)
	goal = np.array(place)*(math.pi/180.0)

	weights = [1.0, 0.0, 0.0]
	T = 25.0

	planner = Planner(2, False)
	planner.replan(start, goal, weights, 0.0, T, 0.5)	
	time.sleep(20)


