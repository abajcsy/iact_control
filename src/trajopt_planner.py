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


#Q2: += pi...
#we can do distance to table well
#we cant seem to do any laptop features/interesting trajectories


#jain/deformation algorithms still needs to be refined



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

		# these variables are for trajopt
		# self.waypts_plan is also treated as the next initial trajectory
		self.waypts_plan = None
		self.num_waypts_plan = None
		self.step_time_plan = None

		# these variables are for the upsampled trajectory
		self.waypts = None
		self.num_waypts = None
		self.step_time = None
		self.waypts_time = None

		self.weights = 1
		self.waypts_prev = None

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
		plotPoint(self.env,self.bodies,[-1.3858/2.0, 0,0],0.2)

		# ---- DEFORMATION Initialization ---- #

		self.alpha = -0.01
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



	# ---- custom feature and cost functions ---- #

	def featurize(self, waypts):
		"""
		computs the user-defined features for a given trajectory.
		input trajectory, output list of feature values
		"""
		features = [None,None]
		features[0] = self.velocity_features(waypts)
		features[1] = [0.0]*(len(waypts)-1)
		for index in range(0,len(waypts)-1):
			features[1][index] = self.laptop_features(waypts[index+1], waypts[index])
		return features
	
	def velocity_features(self, waypts):
		"""
		computes total velocity cost over waypoints, confirmed to match trajopt.
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
		computes the total velocity cost.
		input trajectory, output scalar cost
		"""
		return self.velocity_features(waypts)


	def laptop_features(self, waypt, prev_waypt):

		laptop_xy = np.array([-1.3858/2.0, 0])
		vel_mag = np.linalg.norm(waypt - prev_waypt)
		feature = 0.0

		if len(waypt) < 10:
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]), 1)
			waypt[2] += math.pi
		self.robot.SetDOFValues(waypt)
		coords = robotToCartesian(self.robot)
		EE_coord_xy = coords[6][0:2]
		
		dist = np.linalg.norm(EE_coord_xy - laptop_xy) - 0.2
		if dist < 0:
			feature = -dist#*vel_mag
		return feature

	def laptop_cost(self, waypt):

		prev_waypt = waypt[0:7]
		curr_waypt = waypt[7:14]

		feature = self.laptop_features(curr_waypt, prev_waypt)
		return feature*self.weights




	# ---- custom constraints --- #

	def table_constraint(self, waypt):
		"""

		"""
		if len(waypt) < 10:
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]), 1)
			waypt[2] += math.pi
		self.robot.SetDOFValues(waypt)
		coords = robotToCartesian(self.robot)
		EE_coord_z = coords[6][2]
		if EE_coord_z > 0:
			EE_coord_z = 0
		return -EE_coord_z

	def coffee_constraint(self, waypt):
		if len(waypt) < 10:
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]), 1)
			waypt[2] += math.pi
		self.robot.SetDOFValues(waypt)
		EE_link = self.robot.GetLinks()[7]
		return EE_link.GetTransform()[:2,:3].dot([1,0,0])

	def coffee_constraint_derivative(self, waypt):
		if len(waypt) < 10:
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]), 1)
			waypt[2] += math.pi
		self.robot.SetDOFValues(waypt)
		world_dir = self.robot.GetLinks()[7].GetTransform()[:3,:3].dot([1,0,0])
		return np.array([np.cross(self.robot.GetJoints()[i].GetAxis(), world_dir)[:2] for i in range(7)]).T.copy()



	# ---- here's trajOpt --- #
		
	def trajOpt(self, start, goal):
		"""
		computes a plan from start to goal using optimizer.
		updates the waypts_plan
		"""
		if len(start) < 10:
			aug_start = np.append(start.reshape(7), np.array([0,0,0]), 1)
		self.robot.SetDOFValues(aug_start)

		self.num_waypts_plan = 6
		if self.waypts_plan == None:
			#if no plan, straight line
			init_waypts = np.zeros((self.num_waypts_plan,7))
			init_waypts[0,:] = start
			init_waypts[5,:] = goal
			midpoint = [211.9, 96.0, 205.6, 58.4, 234.8, 292.6, 198.8]
			midpoint_form = np.array(midpoint)*(math.pi/180.0)
			init_waypts[1,:] = start
			init_waypts[2,:] = midpoint_form
			init_waypts[3,:] = midpoint_form
			init_waypts[4,:] = goal
			for count in range(self.num_waypts_plan):
				init_waypts[count,:] = start + count/(self.num_waypts_plan - 1.0)*(goal - start)
		else:
			#if is plan, use previous as initial plan
			init_waypts = self.waypts_plan 
		
		request = {
			"basic_info": {
				"n_steps": self.num_waypts_plan,
				"manip" : "j2s7s300"
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

		# add custom cost functions
		for t in range(1,self.num_waypts_plan): 
			#prob.AddErrorCost(self.laptop_cost, [(t-1,j) for j in range(7)]+[(t,j) for j in range(7)], "HINGE", "laptop%i"%t)
			prob.AddCost(self.laptop_cost, [(t-1,j) for j in range(7)]+[(t,j) for j in range(7)], "laptopC%i"%t)

		#for t in range(1,self.num_waypts_plan - 1):
			#prob.AddConstraint(self.laptop_cost, [(t-1,j) for j in range(7)]+[(t,j) for j in range(7)], "EQ", "up%i"%t)
			#prob.AddConstraint(self.table_constraint, [(t,j) for j in range(7)], "INEQ", "up%i"%t)
			#prob.AddConstraint(self.coffee_constraint, self.coffee_constraint_derivative, [(t,j) for j in range(7)], "EQ", "up%i"%t)

		result = trajoptpy.OptimizeProblem(prob)
		self.waypts_plan = result.GetTraj()
		self.step_time_plan = (self.final_time - self.start_time)/(self.num_waypts_plan - 1)


	# ---- here's our algorithms for modifying the trajectory ---- #

	def learnWeights(self, u_h):
		
		if self.deform(u_h):
			new_features = self.featurize(self.waypts)
			old_features = self.featurize(self.waypts_prev)
			Phi_p = np.array([new_features[0], sum(new_features[1])])
			Phi = np.array([old_features[0], sum(old_features[1])])

			update = Phi_p - Phi
			curr_weight = self.weights - 10.0*update[1]
			if curr_weight > 10.0:
				curr_weight = 10.0
			elif curr_weight < 0.0:
				curr_weight = 0.0

			print "here is the new weight for the table:"
			print curr_weight

			self.weights = curr_weight
			return self.weights


	def deform(self, u_h):
		"""
		deform the next n waypoints of the upsampled trajectory
		updates the upsampled trajectory, stores old trajectory
		"""
		deform_waypt_idx = self.curr_waypt_idx + 1
		if (deform_waypt_idx + self.n) > self.num_waypts:
			return False
		self.waypts_prev = copy.deepcopy(self.waypts)
		gamma = np.zeros((self.n,7))
		for joint in range(7):
			gamma[:,joint] = self.alpha*np.dot(self.H, u_h[joint])
		self.waypts[deform_waypt_idx : self.n + deform_waypt_idx, :] += gamma
		plotTraj(self.env, self.robot, self.bodies, self.waypts, [0, 1, 0])
		return True
	

	# ---- replanning, upsampling, and interpolating ---- #

	def replan(self, start, goal, weights, start_time, final_time, step_time):
		"""
		replan the trajectory from start to goal given weights.
		input trajectory parameters, update raw and upsampled trajectories
		"""
		self.start_time = start_time
		self.final_time = final_time
		self.curr_waypt_idx = 0
		self.weights = weights
		self.trajOpt(start, goal)
		self.upsample(step_time)
		plotTraj(self.env,self.robot,self.bodies,self.waypts, [0, 0, 1])

	def upsample(self, step_time):
		"""
		put waypoints along trajectory at step_time increments.
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
		curr_pos 	7x1 vector of current joint angles (degrees)
		----
		"""

		#TODO: for some reason the 3rd joint is always off by pi in OpenRAVE -- add pi to it as a hack for now
		pos = np.array([curr_pos[0][0],curr_pos[1][0],curr_pos[2][0]+math.pi,curr_pos[3][0],curr_pos[4][0],curr_pos[5][0],curr_pos[6][0],0,0,0])
		
		self.robot.SetDOFValues(pos)

if __name__ == '__main__':

	trajplanner = Planner()

	candlestick = np.array([180.0]*7)	
	home_pos = np.array([103.366,197.13,180.070,43.4309,265.11,257.271,287.9276])

	s = np.array(home_pos)*(math.pi/180.0)
	g = np.array(candlestick)*(math.pi/180.0)
	T = 8.0
	features = None
	weights = [1,1]

	trajplanner.replan(s, g, weights, 0.0, T, 1.0)
	#executePathSim(trajplanner.env,trajplanner.robot,trajplanner.waypts)
	time.sleep(50)

	

	"""
	def coffee_constraint(self, waypt):
		if len(waypt) < 10:
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]), 1)
			waypt[2] += math.pi
		self.robot.SetDOFValues(waypt)
		EE_link = self.robot.GetLinks()[7]
		return EE_link.GetTransform()[:2,:3].dot([1,0,0])

	def coffee_constraint_derivative(self, waypt):
		if len(waypt) < 10:
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]), 1)
			waypt[2] += math.pi
		self.robot.SetDOFValues(waypt)
		world_dir = self.robot.GetLinks()[7].GetTransform()[:3,:3].dot([1,0,0])
		return np.array([np.cross(self.robot.GetJoints()[i].GetAxis(), world_dir)[:2] for i in range(7)]).T.copy()

	def table_features(self, waypt):
		if len(waypt) < 10:
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]), 1)
			waypt[2] += math.pi
		self.robot.SetDOFValues(waypt)
		coords = robotToCartesian(self.robot)
		EEcoord_z = coords[6][2]

		return (1.0 - EEcoord_z)**2
	
	def table_cost(self, waypt):
		prev_waypt = waypt[0:7]
		curr_waypt = waypt[7:14]

		feature = self.table_features(curr_waypt)
		return feature*self.weights[1]*np.linalg.norm(curr_waypt - prev_waypt)

cost = [0.0]*len(coords)
		epsilon = 0.025
		jointIdx = 0
		for coord in coords:
			dist = np.linalg.norm(coord[0:2] - laptop_xy) - 0.15
			if dist < 0:
				cost[jointIdx] = (-dist + 1/(2 * epsilon))*vel_mag
			elif 0 < dist <= epsilon:
				cost[jointIdx] = (1/(2 * epsilon) * (dist - epsilon)**2)*vel_mag
			jointIdx += 1
		return cost

	"""	
