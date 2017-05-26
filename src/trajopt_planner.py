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

import interactpy
from interactpy import *

import logging
import pid
import copy

from catkin.find_in_workspaces import find_in_workspaces

logging.getLogger('prpy.planning.base').addHandler(logging.NullHandler())

#Q2: += pi...

#C1: the openrave perspective could be improved, check our frame
#C2: the rotation in plottable is not in SO(3)?
#C3: add vertical offset to table_feature
#C4: should be a better way of holding the undeformed trajectory

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

		self.weights = [1, 0]
		self.waypts_prev = None

		# ---- OpenRAVE Initialization ---- #
		
		# initialize robot and empty environment
		model_filename = 'jaco_dynamics'
		self.env, self.robot = interact.initialize_empty(model_filename, empty=True)

		# insert any objects you want into environment
		self.bodies = []
	
		# plot the table and table mount
		self.plotTable()
		self.plotTableMount()
		self.plotLaptop()

		# set the perspective
		viewer = self.env.GetViewer()
		viewer.SetSize(1000,1000)
		viewer.SetCamera([
			[0.,  0., -1., 1.],
			[1.,  0.,  0., 0.],
			[0., -1.,  1., 0.],
			[0.,  0.,  0., 1.]])
		viewer.SetBkgndColor([0.8,0.8,0.8])

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
		features = [None, None]
		features[0] = self.velocity_features(waypts)
		features[1] = [0.0]*(len(waypts)-1)
		for index in range(0,len(waypts)-1):
			features[1][index] = self.table_features(waypts[index+1])
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
		feature = self.velocity_features(waypts)
		return feature*self.weights[0]	
	
	def table_features(self, waypt):
		"""
		determines the distance between the end-effector and the table.
		input waypoint, output scalar feature
		"""
		if len(waypt) < 10:
			waypt = np.append(waypt.reshape(7), np.array([0,0,0]), 1)
			waypt[2] += math.pi
		self.robot.SetDOFValues(waypt)
		coords = self.robotToCartesian()
		EEcoord_z = coords[6][2]
		return -EEcoord_z
	
	def table_cost(self, waypt):
		"""
		computs the cost based on distance from end-effector to table.
		input waypoint, output scalar cost
		"""
		feature = self.table_features(waypt)
		return feature*self.weights[1]

	#def laptop_features(self, waypt):
	#	"""
	#	determines the distance between the end-effector and the laptop.
	#	input waypoint, output scalar feature
	#	"""
	#	if len(waypt) < 10:
	#		waypt = np.append(waypt.reshape(7), np.array([0,0,0]), 1)
	#	self.robot.SetDOFValues(waypt)
	#	coords = self.robotToCartesian()
	#	EEcoord_xy = coords[6][0:2]
	#	return -np.linalg.norm(EEcoord_xy - laptop_xy)

	#def laptop_cost(self, waypt):
	#	"""
	#	computs the cost based on distance from end-effector to laptop.
	#	input waypoint, output scalar cost
	#	"""
	#	feature = self.laptop_features(waypt)
	#	return feature*self.weights[2]
	
	
	# ---- here's trajOpt --- #
		
	def trajOpt(self, start, goal):
		"""
		computes a plan from start to goal using optimizer.
		updates the waypts_plan
		"""
		if len(start) < 10:
			aug_start = np.append(start.reshape(7), np.array([0,0,0]), 1)
		self.robot.SetDOFValues(aug_start)

		self.num_waypts_plan = 10
		if self.waypts_plan == None:
			#if no plan, straight line
			init_waypts = np.zeros((self.num_waypts_plan,7))
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
				"params": {"coeffs": [1]}
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
			# use numerical method 
			#prob.AddErrorCost(self.obstacle_cost, [(t,j) for j in range(7)], "ABS", "obstacleC%i"%t)
			prob.AddCost(self.table_cost, [(t,j) for j in range(7)], "obstacleC%i"%t)

		result = trajoptpy.OptimizeProblem(prob)
		self.waypts_plan = result.GetTraj()
		self.step_time_plan = (self.final_time - self.start_time)/(self.num_waypts_plan - 1)


	# ---- here's our algorithms for modifying the trajectory ---- #

	def jainThing(self, u_h):
		
		if self.deform(u_h):
			new_features = self.featurize(self.waypts)
			old_features = self.featurize(self.waypts_prev)
			Phi_p = np.array([new_features[0], sum(new_features[1])])
			Phi = np.array([old_features[0], sum(old_features[1])])

			update = Phi_p - Phi
			print "here is the change in features"
			print "new features: " + str(Phi_p)
			print "old features: " + str(Phi)	
			print "feature diff: " + str(update) 
	
			curr_weight = self.weights[1] - 0.1*update[1]
			if curr_weight > 1:
				curr_weight = 1
			elif curr_weight < 0:
				curr_weight = 0

			print "here is the new weight for table:"
			print curr_weight

			self.weights[1] = curr_weight
			return self.weights
		#replan with 


	def deform(self, u_h):

		deform_waypt_idx = self.curr_waypt_idx + 1
		if (deform_waypt_idx + self.n) > self.num_waypts:
			return False

		self.waypts_prev = copy.deepcopy(self.waypts)

		gamma = np.zeros((self.n,7))
		for joint in range(7):
			gamma[:,joint] = self.alpha*np.dot(self.H, u_h[joint])
		self.waypts[deform_waypt_idx : self.n + deform_waypt_idx, :] += gamma
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
		self.plotTraj()

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


	# ------- Plotting & Conversion Utils ------- #

	def robotToCartesian(self):
		"""
		Converts robot configuration into a list of cartesian 
		(x,y,z) coordinates for each of the robot's links.
		------
		Returns: 7-dimensional list of 3 xyz values
		"""
		links = self.robot.GetLinks()
		cartesian = [None]*7
		i = 0
		for i in range(1,8):
			link = links[i] 
			tf = link.GetTransform()
			cartesian[i-1] = tf[0:3,3]

		return cartesian

	def plotTraj(self):
		"""
		Plots the best trajectory found or planned
		"""

		for waypoint in self.waypts:
			dof = np.append(waypoint, np.array([1, 1, 1]))
			dof[2] += math.pi
			self.robot.SetDOFValues(dof)
			coord = self.robotToCartesian()
			self.plotPoint(coord[6], 0.01)

	def plotPoint(self, coords, size=0.1):
		"""
		Plots a single cube point in OpenRAVE at coords(x,y,z) location
		"""
		with self.env:
			color = np.array([0, 1, 0])

			body = RaveCreateKinBody(self.env, '')
			body.InitFromBoxes(np.array([[coords[0], coords[1], coords[2],
						  size, size, size]]))
			body.SetName("pt"+str(len(self.bodies)))
			self.env.Add(body, True)
			body.GetLinks()[0].GetGeometries()[0].SetDiffuseColor(color)
			self.bodies.append(body)

	def plotTable(self):
		"""
		Plots the robot table in OpenRAVE.
		"""
		# load table into environment
		objects_path = find_in_workspaces(
				project='interactpy',
				path='envdata',
				first_match_only=True)[0]
		self.env.Load('{:s}/table.xml'.format(objects_path))
		table = self.env.GetKinBody('table')
		table.SetTransform(np.array([[0.0, 1.0,  0.0, -0.31], #should be negative 1?
	  								 [1.0, 0.0,  0.0, 0],
				                     [0.0, 0.0,  1.0, -0.932],
				                     [0.0, 0.0,  0.0, 1.0]]))
		color = np.array([0.9, 0.75, 0.75])
		table.GetLinks()[0].GetGeometries()[0].SetDiffuseColor(color)

	def plotTableMount(self):
		"""
		Plots the robot table mount in OpenRAVE.
		"""
		# create robot base attachment
		body = RaveCreateKinBody(self.env, '')
		body.InitFromBoxes(np.array([[0,0,0, 0.14605,0.4001,0.03175]]))
		body.SetTransform(np.array([[1.0, 0.0,  0.0, 0],
				                     [0.0, 1.0,  0.0, 0],
				                     [0.0, 0.0,  1.0, -0.132],
				                     [0.0, 0.0,  0.0, 1.0]]))
		body.SetName("robot_mount")
		self.env.Add(body, True)
		color = np.array([0.9, 0.58, 0])
		body.GetLinks()[0].GetGeometries()[0].SetDiffuseColor(color)
		self.bodies.append(body)

	def plotLaptop(self):
		"""
		Plots the robot table mount in OpenRAVE.
		"""
		# create robot base attachment
		body = RaveCreateKinBody(self.env, '')
		#12 x 9 x 1 in, 0.3048 x 0.2286 x 0.0254 m
		# divide by 2: 0.1524 x 0.1143 x 0.0127
		#20 in from robot base
		body.InitFromBoxes(np.array([[0,0,0,0.1143,0.1524,0.0127]]))
		body.SetTransform(np.array([[1.0, 0.0,  0.0, -0.68],
				                     [0.0, 1.0,  0.0, 0],
				                     [0.0, 0.0,  1.0, -0.132],
				                     [0.0, 0.0,  0.0, 1.0]]))
		body.SetName("laptop")
		self.env.Add(body, True)
		color = np.array([0, 0, 0])
		body.GetLinks()[0].GetGeometries()[0].SetDiffuseColor(color)
		self.bodies.append(body)


	def executePathSim(self):
		"""
		Executes in the planned trajectory in simulation
		"""
		traj = RaveCreateTrajectory(self.env,'')
		traj.Init(self.robot.GetActiveConfigurationSpecification())
		for i in range(len(self.waypts)):
			traj.Insert(i, self.waypts[i])
		self.robot.ExecutePath(traj)

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
	trajplanner.executePathSim()
	time.sleep(20)



	"""
	def D(self, coord, xyz=True):

		obstacle_coords = np.array([-0.508, 0.0, 0.0])
		obstacle_radius = 0.15
		#self.plotPoint([-0.508, 0.0, 0.0], obstacle_radius)

		if xyz is False:
			obstacle_radius = 0.15
			obstacle_coords = np.array([-0.508, 0.0])
			#self.plotPoint([-0.508, 0.0, 0.0], obstacle_radius)
			coord = coord[0:2]

		dist = np.linalg.norm(coord - obstacle_coords) - obstacle_radius

		return dist
	"""
		
	"""
	def obstacle_features7DOF(self, dof):

		if len(dof) < 10:
			padding = np.array([0,0,0])
			dof = np.append(dof.reshape(7), padding, 1)
			dof[2] = dof[2]+math.pi
		self.robot.SetDOFValues(dof)
		coords = self.robotToCartesian()

		cost = [0.0]*len(coords)
		jointIdx = 0
		epsilon = 0.4
		for coord in coords:
			dist = self.D(coord,xyz=True)
			if dist < 0:
				cost[jointIdx] = -dist + 1/(2 * epsilon)
			elif 0 < dist <= epsilon:
				cost[jointIdx] = 1/(2 * epsilon) * (dist - epsilon)**2			
			jointIdx += 1

		return cost
	"""

