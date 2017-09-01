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


HUMAN_TASK = 0
COFFEE_TASK = 1
TABLE_TASK = 2
LAPTOP_TASK = 3

OBS_CENTER = [-1.3858/2.0 - 0.1, -0.1, 0.0]
HUMAN_CENTER = [0.0, 0.2, 0.0]

class Planner(object):
	"""
	This class plans a trajectory from start to goal 
	with TrajOpt. 
	"""

	def __init__(self, task, demo):

		# ---- important internal variables ---- #

		self.task = task
		self.demo = demo
		if self.demo:
			self.MAX_ITER = 40
		else:
			self.MAX_ITER = 40

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

		self.num_f = 3
		self.P_f = [1.0/self.num_f]*self.num_f

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
		
	def trajOpt(self, start, goal):
		"""
		Computes a plan from start to goal using trajectory optimizer.
		Reference: http://joschu.net/docs/trajopt-paper.pdf
		---
		input is start and goal pos, updates the waypts_plan
		"""
		if len(start) < 10:
			aug_start = np.append(start.reshape(7), np.array([0,0,0]), 1)
		self.robot.SetDOFValues(aug_start)

		self.num_waypts_plan = 4
		#if self.waypts_plan == None:
		init_waypts = np.zeros((self.num_waypts_plan,7))
		for count in range(self.num_waypts_plan):
			init_waypts[count,:] = start + count/(self.num_waypts_plan - 1.0)*(goal - start)
		#else:
		#	init_waypts = self.waypts_plan 
		
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

		result = trajoptpy.OptimizeProblem(prob)
		self.waypts_plan = result.GetTraj()
		self.step_time_plan = (self.final_time - self.start_time)/(self.num_waypts_plan - 1)

		bodies = []
		plotTraj(self.env,self.robot,self.bodies,self.waypts_plan, size=10,color=[0, 0, 1])


	# ---- here's our algorithms for modifying the trajectory ---- #

	def Pf(self, f_idx):
		return self.P_f[f_idx]

	def Pdelta(self, delta):
		norm = np.sum([np.exp(-delta[i]) for i in range(self.num_f)])
		return np.sum(([(np.exp(-delta[i])/norm)*self.Pf(i) for i in range(self.num_f)]))
	
	def Pdelta_givenf(self, delta, f_idx):
		partition = np.sum([np.exp(-delta[i]) for i in range(self.num_f)])
		return np.exp(-delta[f_idx])/partition

	def update_Pf(self, delta):
		new_Pf = [0.0]*self.num_f
		for f_idx in range(self.num_f):
			new_Pf[f_idx] = (self.Pdelta_givenf(delta,f_idx)*self.Pf(f_idx))/self.Pdelta(delta)
		self.P_f = new_Pf

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
			update_gains = [2.0, 2.0, 5.0]

			# [max_weight_coffee, max_weight_table, max_weight_laptop] 
			max_weights = [1.0, 1.0, 5.0] 

			update = Phi_p - Phi
			which_update = 2 # 1 = update all, 2 = update with max, 3 = update with max P(feature)

			print "Phi prev: " + str(Phi_p)
			print "Phi curr: " + str(Phi)
			print "Phi_p - Phi = " + str(update)
				
			if which_update is 1:
				# update all weights 
				curr_weight = [self.weights[0] - update_gains[0]*update[1], self.weights[1] - update_gains[1]*update[2], self.weights[2] - update_gains[2]*update[3]]		
			elif which_update is 2:
				change_in_features = [update[1], update[2], update[3]]

				# get index of maximal change
				max_idx = np.argmax(np.fabs(change_in_features))
				
				# update only weight of feature with maximal change
				curr_weight = [self.weights[0], self.weights[1], self.weights[2]]
				curr_weight[max_idx] = curr_weight[max_idx] - update_gains[max_idx]*change_in_features[max_idx]
			else:
				delta = [update[1], update[2], update[3]]

				# update probability distribution given delta in features
				print "P_f before: " + str(self.P_f)
				self.update_Pf(delta)
				print "P_f after: " + str(self.P_f)
				
				# get index of maximal likelihood feature
				max_idx = np.argmax(self.P_f)
				
				# update only weight of feature with maximal change
				curr_weight = [self.weights[0], self.weights[1], self.weights[2]]
				curr_weight[max_idx] = curr_weight[max_idx] - update_gains[max_idx]*delta[max_idx]

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

	def replan(self, start, goal, weights, start_time, final_time, step_time):
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
		self.trajOpt(start, goal)
		#print "waypts_plan after trajopt: " + str(self.waypts_plan)
		self.upsample(step_time)
		#print "waypts_plan after upsampling: " + str(self.waypts_plan)
		#plotTraj(self.env,self.robot,self.bodies,self.waypts_plan, [0, 0, 1])

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

	time.sleep(50)



"""
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


HUMAN_TASK = 0
COFFEE_TASK = 1
TABLE_TASK = 2
LAPTOP_TASK = 3

MAX_ITER = 2
LAPTOP_xyz = [-1.3858/2.0 - 0.1, -0.1, 0.0]
LAPTOP_xy = np.array(LAPTOP_xyz[0:2])
HUMAN_xyz = [0.0, 0.3, 0.0]
HUMAN_xy = np.array(HUMAN_xyz[0:2])



class Planner(object):


	def __init__(self, task, demo):

		self.task = task
		self.demo = demo

		# ---- DEFORMATION Initialization ---- #

		self.alpha = -0.1
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
		if self.task == TABLE_TASK or self.task == COFFEE_TASK:
			self.num_waypts_plan = 6
		else:
			self.num_waypts_plan = 4

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
		#plotSphere(self.env,self.bodies,LAPTOP_xyz,0.4)
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


	# -- Keep Coffee Upright -- #

	def coffee_features(self, waypt):

		if len(waypt) < 10:
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]), 1)
			waypt[2] += math.pi
		self.robot.SetDOFValues(waypt)
		EE_link = self.robot.GetLinks()[7]
		return sum(abs(EE_link.GetTransform()[:2,:3].dot([1,0,0])))

	def coffee_cost(self, waypt):

		feature = self.coffee_features(waypt)
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
		coords = robotToCartesian(self.robot)
		EE_coord_xy = coords[6][0:2]
		dist = np.linalg.norm(EE_coord_xy - LAPTOP_xy) - 0.4
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
		coords = robotToCartesian(self.robot)
		EE_coord_xy = coords[6][0:2]
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

		if self.demo:
			max_iter = 40
		else:
			max_iter = MAX_ITER		

		request = {
			"basic_info": {
				"n_steps": self.num_waypts_plan,
				"manip" : "j2s7s300",
				"max_iter": max_iter
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
			if self.task == TABLE_TASK:
				prob.AddCost(self.table_cost, [(t,j) for j in range(7)], "table%i"%t)
			elif self.task == COFFEE_TASK:
				prob.AddCost(self.coffee_cost, [(t,j) for j in range(7)], "coffee%i"%t)
			elif self.task == LAPTOP_TASK:
				prob.AddCost(self.laptop_cost, [(t-1,j) for j in range(7)]+[(t,j) for j in range(7)], "laptop%i"%t)
			elif self.task == HUMAN_TASK:
				prob.AddCost(self.human_cost, [(t-1,j) for j in range(7)]+[(t,j) for j in range(7)], "human%i"%t)
	
		for t in range(1,self.num_waypts_plan - 1):
			prob.AddConstraint(self.table_constraint, [(t,j) for j in range(7)], "INEQ", "up%i"%t)
		
		result = trajoptpy.OptimizeProblem(prob)
		self.waypts = result.GetTraj()

		#self.bodies = []
		#plotTraj(self.env,self.robot,self.bodies,self.waypts, size=0.05, color=[0, 0, 1])




	# ---- update rule for weights ---- #

	def update(self, waypts_deform, waypts_prev):

		Phi_p = 0.0
		Phi = 0.0
		if self.task == TABLE_TASK:
			for count in range(1,waypts_deform.shape[0]):
				Phi_p += self.table_features(waypts_deform[count,:])
				Phi += self.table_features(waypts_prev[count,:])
			self.weights = max([0.0,self.weights - 0.2*(Phi_p - Phi)])
			return self.weights
		elif self.task == COFFEE_TASK:
			for count in range(1,waypts_deform.shape[0]):
				Phi_p += self.coffee_features(waypts_deform[count,:])
				Phi += self.coffee_features(waypts_prev[count,:])
			self.weights = max([0.0,self.weights - 0.2*(Phi_p - Phi)])
			return self.weights
		elif self.task == LAPTOP_TASK:
			for count in range(1,waypts_deform.shape[0]):
				Phi_p += self.laptop_features(waypts_deform[count,:], waypts_deform[count-1,:])
				Phi += self.laptop_features(waypts_prev[count,:], waypts_prev[count-1,:])
			self.weights = - 10.0*(Phi_p - Phi)
			return self.weights
		elif self.task == HUMAN_TASK:
			for count in range(1,waypts_deform.shape[0]):
				Phi_p += self.human_features(waypts_deform[count,:], waypts_deform[count-1,:])
				Phi += self.human_features(waypts_prev[count,:], waypts_prev[count-1,:])
			self.weights = max([0.0,self.weights - 10.0*(Phi_p - Phi)])
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
		print waypts_deform - waypts_prev
		return (waypts_deform, waypts_prev)



	# ---- obtaining the desired position ---- #

	def replan(self, start, goal, start_time, final_time, weights):

		if weights == None:
			return
		self.weights = weights
		self.trajOpt(start, goal, start_time, final_time)
		self.start_time = start_time
		self.final_time = final_time
		self.step_time = (self.final_time - self.start_time)/(self.num_waypts_plan - 1.0)


	def upsample(self, step_time):

		num_waypts = int(math.ceil((self.final_time - self.start_time)/step_time)+0.001) + 1
		waypts = np.zeros((num_waypts,7))
		waypts[0,:] = self.waypts[0,:]
		for count in range(1, num_waypts):
			curr_time = self.start_time + count*step_time
			waypts[count,:] = self.interpolate(curr_time).reshape((1,7))
		self.step_time = step_time
		self.waypts = waypts
			

	def updateStart(self, start, elapsed_time):

		self.start_time += elapsed_time
		self.final_time += elapsed_time
		delta = start - self.waypts[0,:]
		if np.linalg.norm(delta) < 1e-3:
			return
		for count in range(self.num_waypts_plan):
			curr = self.waypts[count,:]
			self.waypts[count,:] = curr + (self.num_waypts_plan - 1.0 - count)/(self.num_waypts_plan - 1.0)*delta


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

		self.env.Destroy()
		RaveDestroy()

	

if __name__ == '__main__':

	time.sleep(50)
"""

"""
	def updateStart(self, start, elapsed_time):

		self.start_time += elapsed_time
		self.final_time += elapsed_time
		delta = start - self.waypts[0,:]
		if np.linalg.norm(delta) < 1e-3:
			return
		for count in range(self.num_waypts_plan):
			curr = self.waypts[count,:]
			self.waypts[count,:] = curr + (self.num_waypts_plan - 1.0 - count)/(self.num_waypts_plan - 1.0)*delta
"""




"""
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

if self.task == MIRROR_TASK:
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
					"type" : "pose", 
					"params" : {"xyz" : [-0.41992156, 0.52793478, 0.57166576], 
							"wxyz" : [1,0,0,0],
							"link": "j2s7s300_link_7",
							"rot_coeffs" : [0,0,0],
							"pos_coeffs" : [10,10,10]
							}
				}			  
				],
				"init_info": {
		            "type": "given_traj",
		            "data": init_waypts.tolist()
				}
			}
"""

