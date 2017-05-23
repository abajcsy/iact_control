import numpy as np
from numpy import linalg
from numpy import linspace
import time
import math

from sympy import symbols
from sympy import lambdify

import trajoptpy
import or_trajopt

import interactpy
from interactpy import *
from util import *

import logging
import pid
import copy

import sim_robot

class Trajectory(object):
	"""
	This class represents a trajectory from start to goal.

	Required parameters:
		waypts   - untimed raw waypoints from Planner
		T 		 - total time for trajectory execution
		features - list of trajectory features
		weights  - list of weights for trajectory features
	"""

	def __init__(self, waypts, weights, T):	

		# raw waypoints from Planner
		self.traj_pts = waypts
		self.num_traj_pts = waypts.shape[0]

		# using the raw waypoints, retime the trajectory to get fine-grained
		# sampled waypoints along the trajectory
		#self.traj = self.waypts
		#self.traj_pts = self.sample_waypoints(self.traj)
		#self.num_traj_pts = len(self.traj_pts)

		# store index of the closest waypoint robot is at
		self.curr_waypt_idx = 0

		# compute time, T_i for each of the waypoints i in {1,...n}
		self.totalT = T
		self.wayptsT = [None]*self.num_traj_pts
		T_sum = 0.0
		if self.num_traj_pts >= 2:
			for i in range(self.num_traj_pts):
				self.wayptsT[i] = T_sum
				print "In plan...T_sum: " + str(T_sum)
				T_sum += self.totalT/(self.num_traj_pts-1)
		else:
			self.wayptsT[0] = self.totalT
	
		self.s = self.traj_pts[0]
		self.g = self.traj_pts[self.num_traj_pts-1]
		self.weight = weights

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

		# create R matrix, and R^-1 matrix
		self.R = np.dot(self.A.T, self.A)
		Rinv = np.linalg.inv(self.R)

		Uh = np.zeros((self.n, 1))
		Uh[0] = 1

		self.H = np.dot(Rinv,Uh)*(np.sqrt(self.n)/np.linalg.norm(np.dot(Rinv,Uh)))

	def deform_propogate(self, u_h):
		"""
		Deforms waypoints given human applied force, u_h, in c-space.
		----
		u_h 	7x1 vector of applied torques
		----
		"""
		# arbitration parameter
		mu = -0.005

		# sanity check - if there are less than n waypoints remaining in 
		# entire trajectory, then just change as many as you can till end
		if (self.curr_waypt_idx + self.n) >= self.num_traj_pts:
			print "Less than n=" + str(self.n) + " waypoints remaining!"
			return

		traj_prev = self.traj_pts

		gamma = np.zeros((self.n,7))
		for joint in range(7):
			gamma[:,joint] = mu*np.dot(self.H, u_h[joint])

		gamma_prev = traj_prev[self.curr_waypt_idx : self.n + self.curr_waypt_idx, :]
		traj_prev[self.curr_waypt_idx : self.n+self.curr_waypt_idx, :] = gamma_prev + gamma

		self.traj_pts = traj_prev

	def deform_waypt(self, u_h):
		"""
		Deform a single waypoint given u_h
		"""
		# arbitration parameter
		mu = -0.005

		if (self.curr_waypt_idx + 1) >= self.num_traj_pts:
			print "Less than 1 waypoints remaining!"
			return

		traj_pts_prime = copy.deepcopy(self.traj_pts)
		
		gamma = np.zeros((1,7))
		for joint in range(7):
			gamma[:,joint] = mu*u_h[joint]

		gamma_prev = traj_pts_prime[self.curr_waypt_idx : 1 + self.curr_waypt_idx, :]
		traj_pts_prime[self.curr_waypt_idx : 1 + self.curr_waypt_idx, :] = gamma_prev + gamma

		weights_prime = update_weights(self.traj_pts, traj_pts_prime)

		return weights_prime

	def interpolate(self, t):
		"""
		Gets the next desired position along trajectory
		by interpolating between waypoints given the current t.
		"""
		target_pos = self.traj_pts[0]
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

		target_pos = np.array(target_pos).reshape((7,1))
		return target_pos


	def featurize(self, traj, totalT):
		"""
		Returns an array of feature values along traj
		using specified features 
		"""
		# TODO
		return 

	def update_weights(self, oldTraj, newTraj):
		"""
		Returns updated weights for new trajectory 
		given old trajectory
		"""

		
		#TODO
		return

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
		Returns: 2D array of configurations along trajectory for every 0.5 seconds 
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

if __name__ == '__main__':
	
	candlestick = np.array([180.0]*7)	
	home_pos = np.array([103.366,197.13,180.070,43.4309,265.11,257.271,287.9276])

	s = np.array(home_pos)*(math.pi/180.0)
	g = np.array(candlestick)*(math.pi/180.0)

	# test deformations
	u_h = np.array([0, 20, 0, 20, 20, 0, 0]).reshape((7,1))
	
