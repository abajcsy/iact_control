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
	This class plans a trajectory from start to goal 
	with TrajOpt. 
	"""

	def __init__(self):	

		# ---- OpenRAVE Initialization ---- #
		
		# initialize openrave and compute waypts
		model_filename = 'jaco_dynamics'
		# model_filename = 'jaco'
		self.env, self.robot = interact.initialize_empty(model_filename, empty=True)

		# insert any objects you want into environment
		self.bodies = []
		coords = [0.5, 0.3, 0.8]
		#self.plotPoint(coords)

		viewer = self.env.GetViewer()

		viewer.SetSize(1000,1000)
		viewer.SetCamera([
			[0.,  0., -1., 1.],
			[1.,  0.,  0., 0.],
			[0., -1.,  1., 0.],
			[0.,  0.,  0., 1.]])


		viewer.SetBkgndColor([0.8,0.8,0.8])

	def plan(self, start, goal, features, weights, T):
		"""
		Computes a plan from newStart to self.g taking T total time.
		"""
		if len(start) < 10:
			start = start.reshape(7)
			print "length too short " + str(len(start))
			padding = np.array([0,0,0])
			start = np.append(start, padding, 1)

		self.robot.SetDOFValues(start)
		orig_ee = self.robot.arm.hand.GetTransform()

		# get the raw waypoints from trajopt		
		# TODO convert features & weights to traj_costs
		# TODO waypts = self.robot.arm.PlanToConfiguration(goal, traj_costs=traj_costs)
		self.waypts = self.robot.arm.PlanToConfiguration(goal)

		return self.waypts


	# ------- Plotting Utils ------- #

	def plotTraj(self):
		"""
		Plots the best trajectory found or planned

		TODO: traj_pts plotting is deprecated!
		"""
		for body in self.bodies:
			self.env.Remove(body)

		self.bodies += plotWaypoints(self.env, self.robot, self.traj_pts)

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
