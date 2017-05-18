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

import sim_robot

logging.getLogger('prpy.planning.base').addHandler(logging.NullHandler())

class Planner(object):
	"""
	This class represents plans a trajectory from start to goal 
	with TrajOpt. 

	Required parameters:
		start    - start cofiguration (at t = 0)
		goal	 - goal configuration (at t = T)
		T 		 - total time for trajectory execution
	"""

	def __init__(self, start, goal, T):	
		# set first and last waypt as start and goal
		# NOTE: assume that start and goal are in radians

		# have to convert convert from 7x1 np matrix to 1x10 np matrix 
		# so need to add padding to start
		padding = np.array([0,0,0])
		start = np.append(start, padding, 1)

		self.s = start
		self.g = goal
		self.totalT = T
		self.curr_pos = self.s

		print "totalT: " + str(self.totalT)

		# ---- OpenRAVE Initialization ---- #
		
		# initialize openrave and compute waypts
		model_filename = 'jaco_dynamics'
		#model_filename = 'jaco'
		self.env, self.robot = interact.initialize_empty(model_filename, empty=True)
		physics = RaveCreatePhysicsEngine(self.env,'ode')
		self.env.SetPhysicsEngine(physics)

		# insert any objects you want into environment
		self.bodies = []
		coords = [0.5, 0.3, 0.8]
		self.plotPoint(coords)

		viewer = self.env.GetViewer()

		viewer.SetSize(1000,1000)
		"""
		viewer.SetCamera([[ 0.94684722, -0.12076704,  0.29815376,  0.21004671],
			   [-0.3208323 , -0.42191214,  0.84797216, -1.40675116],
			   [ 0.0233876 , -0.89855744, -0.43823231,  1.04685986],
			   [ 0.        ,  0.        ,  0.        ,  1.        ]])
		"""

		viewer.SetCamera([
			[0.,  0., -1., 1.],
			[1.,  0.,  0., 0.],
			[0., -1.,  1., 0.],
			[0.,  0.,  0., 1.]])


		viewer.SetBkgndColor([0.8,0.8,0.8])

		# --------------------------------- #

		# plan trajectory from self.s to self.g with self.totalT time
		self.plan(self.s, self.totalT)
		print "Trajectory waypoint times T:" + str(self.wayptsT) 

		# plot the trajectory
		self.plotTraj()

		# ---- DEFORMATION Initialization ---- #
		self.n = 5 # number of waypoints that will be deformed

		# create A matrix
		self.A = np.zeros((self.n+2, self.n)) 
		# fill diagonal with 1's
		np.fill_diagonal(self.A, 1)
		# populate off diagonal with -2 and twice off diag with 1
		for i in range(self.n):
			self.A[i+1][i] = -2
			self.A[i+2][i] = 1

		print "A: " + str(self.A)

		# create R matrix, and R^-1 matrix
		self.R = np.dot(self.A.T, self.A)
		Rinv = np.linalg.inv(self.R)
		print "R: " + str(self.R)

		Uh = np.zeros((self.n, 1))
		Uh[0] = 1

		self.H = np.dot(Rinv,Uh)*(np.sqrt(self.n)/np.linalg.norm(np.dot(Rinv,Uh)))
		print "H: " + str(self.H)
		# --------------------------------- #

		# store cartesian waypoints for plotting and debugging
		#self.cartesian_waypts = self.get_cartesian_waypts()
		#self.plot_cartesian_waypts(self.cartesian_waypts)

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

	def plan(self, newStart, T):
		"""
		Computes a plan from newStart to self.g taking T total time.
		"""
		if len(newStart) < 10:
			newStart = newStart.reshape(7)
			print "length too short " + str(len(newStart))
			padding = np.array([0,0,0])
			print "start: " + str(newStart)
			print "padding: " + str(padding)
			newStart = np.append(newStart, padding, 1)

		self.s = newStart
		self.totalT = T

		print "in plan...totalT: " + str(self.totalT)
		self.robot.SetDOFValues(self.s)
		orig_ee = self.robot.arm.hand.GetTransform()

		# get the raw waypoints from trajopt		
		self.waypts = self.robot.arm.PlanToConfiguration(self.g)
		self.num_waypts = self.waypts.GetNumWaypoints()
		print "Number of ORIGINAL waypoints: " + str(self.num_waypts)

		# using the raw waypoints, retime the trajectory to get fine-grained
		# sampled waypoints along the trajectory
		self.traj = self.waypts
		self.traj_pts = self.sample_waypoints(self.traj)
		self.num_traj_pts = len(self.traj_pts)

		# store index of the closest waypoint robot is at
		self.curr_waypt_idx = 0

		print self.traj_pts
		print "Number of RESAMPLED waypoints: " + str(self.num_traj_pts)

		#print "selfs: " + str(self.s[:7])
		#print "selfg: " + str(self.g)
		#if np.array_equal(self.s[:7], self.g):
		#	print "START AND GOAL ARE THE SAME. Just holding position."

		#for i in range(self.num_waypts):
		#	print self.waypts.GetWaypoint(i)

		# compute time, T_i for each of the waypoints i in {1,...n}
		self.wayptsT = [None]*self.num_traj_pts
		T_sum = 0.0
		if self.num_traj_pts >= 2:
			for i in range(self.num_traj_pts):
				self.wayptsT[i] = T_sum
				print "In pl98an...T_sum: " + str(T_sum)
				T_sum += self.totalT/(self.num_traj_pts-1)
		else:
			self.wayptsT[0] = self.totalT

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
			#print "cartesian " + str(i) + ": "  + str(cartesian)
			cartesian.append(tf)
		#print "cartesian: " + str(cartesian)
		return np.array(cartesian)

	def plot_cartesian_waypts(self, cartesian):
		"""
		Plots cartesian waypoints in OpenRAVE.
		"""
		for i in range(self.num_waypts):
			self.plotPoint(cartesian[i])

	def get_robot(self):
		"""
		Returns robot model
		"""
		return self.robot

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

	def deform(self, u_h):
		"""
		Deforms waypoints given human applied force, u_h, in c-space.
		----
		u_h 	7x1 vector of applied torques
		----
		"""
		# print "IN deform(u_h): "

		#TODO ONLY WORKS FOR INTERACTION WITH ONE DOF
		# grab the force applied to one dof
		#force_h = 0.0
		#dof = 0
		#for i in range(len(u_h)):
		#	if u_h[i][0] != 0:
		#		force_h = u_h[i][0]
		#		dof = i

		# TODO THIS IS TEMPORARY NEGATION TO GET CORRECT DIRECTION
		#force_h = -force_h

		#print "--force_h: " + str(force_h)
		#print "--dof: " + str(dof)

		# arbitration parameter
		mu = -0.005
		#print "--mu: " + str(mu)
		# current waypoint idx
		i = self.curr_waypt_idx
		#print "--i: " + str(i)

		# sanity check - if there are less than n waypoints remaining in 
		# entire trajectory, then just change as many as you can till end
		if (self.curr_waypt_idx + self.n) > self.num_traj_pts:
			print "Less than n=" + str(self.n) + " waypoints remaining!"
			return

		traj_prev = self.traj_pts

		gamma = np.zeros((self.n,7))
		for ii in range(7):
			gamma[:,ii] = mu*np.dot(self.H, u_h[ii])

		traj_prev_tmp = copy.deepcopy(traj_prev)

		gamma_prev = traj_prev[i : self.n + i, :]
		traj_prev[i : self.n+i, :] = gamma_prev + gamma

		self.traj_pts = traj_prev

		# plot the trajectory
		self.plotTraj()

	def interpolate(self, t):
		"""
		Gets the next desired position along trajectory
		by interpolating between waypoints given the current t.
		"""
		target_pos = self.traj_pts[0]
		print "NUMBER WAYPOINTS: " + str(self.num_traj_pts)
		# TODO CHECK CORNER CASES START/END	
		if self.num_traj_pts >= 2:
			for i in range(self.num_traj_pts-1):
				# if between two waypoints, interpolate
				if t > self.wayptsT[i] and t < self.wayptsT[i+1]:
					print "between waypt " + str(i) + " and " + str(i+1)
					# linearly interpolate between waypts
					prev = self.traj_pts[i]
					next = self.traj_pts[i+1]
					ti = self.wayptsT[i]
					ti1 = self.wayptsT[i+1]
					deltaT = ti1 - ti
					theta = (next - prev)*((t-ti)/deltaT) + prev
					target_pos = theta
					# store index of next waypoint that robot is near
					self.curr_waypt_idx = i+1
				# if exactly at a waypoint, return that waypoint
				elif t == self.wayptsT[i]:
					target_pos = self.traj_pts[i]		
					# store index of current waypoint that robot is near
					self.curr_waypt_idx = i
		else:
			print "ONLY ONE WAYPT, CAN'T INTERPOLATE."
	
		# if times up, just go to goal
		if t > self.totalT:
			print "TIME IS UP. GOING TO FINAL WAYPOINT."
			target_pos = self.traj_pts[self.num_traj_pts-1]	
			# store index of current waypoint that robot is near
			self.curr_waypt_idx = self.num_traj_pts-1

		# plot the target position in task-space
		dof = np.append(target_pos, np.array([1, 1, 1]))
		coord = transformToCartesian(dofToTransform(self.robot, dof))
		self.plotPoint(coord, size=0.01)

		target_pos = np.array(target_pos).reshape((7,1))
		return target_pos

	def interpolate_waypts(self, t):
		"""
		Gets the next desired position along trajectory
		by interpolating between waypoints given the current t.
		"""
		target_pos = self.waypts.GetWaypoint(0)
		print "NUMBER WAYPOINTS: " + str(self.num_waypts)
		if self.num_waypts >= 2:
			for i in range(self.num_waypts-1):
				# if between two waypoints, interpolate
				if t > self.wayptsT[i] and t < self.wayptsT[i+1]:
					print "between waypt " + str(i) + " and " + str(i+1)
					# linearly interpolate between waypts
					prev = self.waypts.GetWaypoint(i)
					next = self.waypts.GetWaypoint(i+1)
					Tprev = self.wayptsT[i]
					Tnext = self.wayptsT[i+1]
					deltaT = Tnext - Tprev
					theta = (next - prev)*(1/deltaT)*t + prev
					target_pos = theta
				# if exactly at a waypoint, return that waypoint
				elif t == self.wayptsT[i]:
					target_pos = self.waypts.GetWaypoint(i)			
		else:
			print "ONLY ONE WAYPT, CAN'T INTERPOLATE."
	
		# if times up, just go to goal
		if t > self.totalT:
			print "TIME IS UP. GOING TO FINAL WAYPOINT."
			target_pos = self.waypts.GetWaypoint(self.num_waypts-1)	
		target_pos = np.array(target_pos).reshape((7,1))
		return target_pos

	def sample_traj(self, t):
		"""
		Samples from trajectory (continuous) at given time t.
		"""
		sample = self.traj.Sample(t)[:7]
		target_pos = np.array(sample).reshape((7,1))
		return target_pos

	def sample_waypoints(self, traj):
		"""
		Samples waypoints every 0.5 seconds along the trajectory
		Parameters
		----------
		traj : OpenRAVE trajectory
		Returns
		-------
		2D-array of DOFs
		"""

		if traj.GetDuration() == 0:
			print "Retiming trajectory..."
			planningutils.RetimeTrajectory(traj)
		duration = traj.GetDuration()
		dofs = traj.SamplePoints2D(np.append(np.arange(0, duration, 0.5), duration))

		# for some reason, rrt creates dofs of dimension 15????
		dofs = dofs[:, :7]

		return dofs

	def plotTraj(self):
		"""
		Plots the best trajectory found or planned
		"""
		for body in self.bodies:
			self.env.Remove(body)

		self.bodies += plotWaypoints(self.env, self.robot, self.traj_pts)

if __name__ == '__main__':
	
	home = np.array([103.366,197.13,180.070,43.4309,265.11,257.271,287.9276])
	goal = np.array([338.680, 172.142, 25.755, 96.798, 180.497, 137.340, 186.655])
	candlestick = np.array([180.0]*7)	
	
	waypt1 = np.array([136.886, 200.805, 64.022, 116.637, 138.328, 122.469, 179.861])
	waypt2 = np.array([271.091, 225.708, 20.548, 158.572, 160.879, 183.520, 186.644])
	waypt3 = np.array([338.680, 172.142, 25.755, 96.798, 180.497, 137.340, 186.655])

	home_pos = home
	goal_pos = goal

	s = np.array(home_pos)*(math.pi/180.0)
	g = np.array(candlestick)*(math.pi/180.0)

	#robot_starting_dofs = np.array([-1, 2, 0, 2, 0, 4, 0])
	#goal = np.array([0,  2.9 ,  0.0 ,  2.1 ,  0. ,  4. ,  0.])

	#s = robot_starting_dofs
	#g = goal

	print s
	print g

	T = 8.0
	trajplanner = Planner(s,g,T)
	u_h = np.array([0, 20, 0, 20, 20, 0, 0]).reshape((7,1))
	trajplanner.deform(u_h)
	t = 1.0
	theta = trajplanner.interpolate(t)
	newStart = np.array(waypt1).reshape((7,1))
	trajplanner.plan(newStart,10)
	#print "theta: " + str(theta)
	#print trajplanner.traj.Sample(0.1)
	#trajplanner.execute_path_sim()
	time.sleep(20)
