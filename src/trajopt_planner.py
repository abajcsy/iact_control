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

import logging
import pid
import copy
import json

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

		self.weights = None
		self.waypts_prev = None

		# ---- OpenRAVE Initialization ---- #
		
		# initialize openrave and compute waypts
		model_filename = 'jaco_dynamics'
		# model_filename = 'jaco'
		self.env, self.robot = interact.initialize_empty(model_filename, empty=True)

		# insert any objects you want into environment
		self.bodies = []
	
		# plot the table and table mount
		self.plotTable()
		self.plotTableMount()

		# plot obstacles
		coords = [0.2, 0.2, 0.8]
		self.plotPoint(coords, 0.2)
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

		self.alpha = -0.005
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



	# ---- featurization functions ---- #

	def featurize(self, waypts):

		features = [None, None]

		features[0] = self.velocity_features(waypts)

		features[1] = [0.0]*(len(waypts)-1)
		for index in range(1,len(waypts)):
			dof = waypts[index]
			features[1][index-1] = sum(self.obstacle_features(dof))

		return features


	# ---- custom cost functions ---- #

	def D(self, coord):
		"""
		Computes euclidian distance from current coord = (x,y,z)
		to a circular obstacle.
		"""

		obstacle_coords = np.array([0.2, 0.2, 0.8])
		obstacle_radius = 0.2

		dist = np.linalg.norm(coord - obstacle_coords) - obstacle_radius

		return dist


	def obstacle_cost(self, dof):
		"""
		Obstacle cost function that penalizes the robot for being near obstacles.
		From CHOMP algorithm (Zucker, 2013).
		"""
		
		cost = self.obstacle_features(dof)
		for jointIdx in range(7):
			cost[jointIdx] *= self.weights[1]
		return cost


	def obstacle_features(self, dof):

		if len(dof) < 10:
			padding = np.array([0,0,0])
			dof = np.append(dof.reshape(7), padding, 1)
		self.robot.SetDOFValues(dof)
		coords = self.robotToCartesian()

		cost = [0.0]*len(coords)
		jointIdx = 0
		epsilon = 0.4
		for coord in coords:
			dist = self.D(coord)
			if dist < 0:
				cost[jointIdx] = -dist + 1/(2 * epsilon)
			elif 0 < dist <= epsilon:
				cost[jointIdx] = 1/(2 * epsilon) * (dist - epsilon)**2			
			jointIdx += 1

		return cost


	def velocity_features(self, waypts):
		"""
		Computes total velocity cost over waypoints.
		Returns scalar cost. 
		"""
		vel = 0.0
		for i in range(1,len(waypts)):
			curr = waypts[i]
			prev = waypts[i-1]
			vel += np.linalg.norm(curr - prev)**2
			
		return vel




	# ---- let's replan a new trajectory ---- #

	def replan(self, start, goal, weights, start_time, final_time, step_time):

		self.start_time = start_time
		self.final_time = final_time
		self.curr_waypt_idx = 0
		self.weights = weights
		self.trajOpt(start, goal)
		self.upsample(step_time)

	def upsample(self, step_time):

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

	def trajOpt(self, start, goal):
		"""
		Computes a plan from start to goal taking T total time.
		"""

		if len(start) < 10:
			padding = np.array([0,0,0])
			aug_start = np.append(start.reshape(7), padding, 1)
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
				"params": {"coeffs": [self.weights[0]]}
			}
			,
			#{
			#	"type": "collision",
			#	"params": {
			#	"coeffs": [w_collision],
			#	"dist_pen": [0.5]
			#	},
			#}
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
			prob.AddErrorCost(self.obstacle_cost, [(t,j) for j in range(7)], "ABS", "obstacleC%i"%t)

		result = trajoptpy.OptimizeProblem(prob)
		self.waypts_plan = result.GetTraj()
		self.step_time_plan = (self.final_time - self.start_time)/(self.num_waypts_plan - 1)



	# ---- let's find the target position ---- #

	def interpolate(self, curr_time):
		"""
		Gets the next desired position along trajectory
		by interpolating between waypoints given the current t.
		"""

		if curr_time >= self.final_time:

			self.curr_waypt_idx = self.num_waypts - 1
			target_pos = self.waypt[self.curr_waypt_idx]

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





	# ---- let's deform the trajectory ---- #

	def jainThing(self, u_h):
		
		if self.deform(u_h):
			new_features = self.featurize(self.waypts)
			old_features = self.featurize(self.waypts_prev)
			Phi_p = [new_features[0], sum(new_features[1])]
			Phi = [old_features[0], sum(old_features[1])]
			print "here is the change in features"
			print Phi_p
			print Phi
			#print new_features - new_features
			
		


	def deform(self, u_h):

		if (self.curr_waypt_idx + self.n) >= self.num_waypts:
			return False

		self.waypts_prev = copy.deepcopy(self.waypts)

		gamma = np.zeros((self.n,7))
		for joint in range(7):
			gamma[:,joint] = self.alpha*np.dot(self.H, u_h[joint])

		gamma_prev = self.waypts[self.curr_waypt_idx : self.n + self.curr_waypt_idx, :]
		self.waypts[self.curr_waypt_idx : self.n + self.curr_waypt_idx, :] = gamma_prev + gamma
		return True





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
		table.SetTransform(np.array([[1.0, 0.0,  0.0, 0],
				                     [0.0, 1.0,  0.0, 0],
				                     [0.0, 0.0,  1.0, -0.832],
				                     [0.0, 0.0,  0.0, 1.0]]))
		color = np.array([0.9, 0.75, 0.75])
		table.GetLinks()[0].GetGeometries()[0].SetDiffuseColor(color)

	def plotTableMount(self):
		"""
		Plots the robot table mount in OpenRAVE.
		"""
		# create robot base attachment
		body = RaveCreateKinBody(env, '')
		body.InitFromBoxes(np.array([[0,0,0, 0.14605,0.4001,0.03175]]))
		body.SetTransform(np.array([[1.0, 0.0,  0.0, 0],
				                     [0.0, 1.0,  0.0, 0],
				                     [0.0, 0.0,  1.0, -0.032],
				                     [0.0, 0.0,  0.0, 1.0]]))
		body.SetName("robot_mount")
		env.Add(body, True)
		color = np.array([0.9, 0.58, 0])
		body.GetLinks()[0].GetGeometries()[0].SetDiffuseColor(color)
		self.bodies.append(body)

	def executePathSim(self):
		"""
		Executes in the planned trajectory in simulation
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
	weights = [1,1]

	trajplanner.replan(s, g, weights, 0.0, T, 1.0)
	trajplanner.executePathSim()
	time.sleep(20)



