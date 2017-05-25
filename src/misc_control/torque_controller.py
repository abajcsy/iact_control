#! /usr/bin/env python
"""
A simple torque-controller class.

Author: Andrea Bajcsy (abajcsy@eecs.berkeley.edu)
"""
import roslib; roslib.load_manifest('kinova_demo')

import time
import math
from openravepy import *

import interactpy
from interactpy import initialize

import numpy as np
from numpy import linalg

class TorqueController(object): 
	"""
	This class represents a node that moves the Jaco with torque-control.
	The joint torques are computed as:

	T = M(theta)*ddtheta + C(theta, dtheta)dtheta + G(theta)

	where:

	M = moment of inertia matrix
	C = coriolis forces matrix
	G = gravity matrix
	theta = joint positions
	dtheta = joint velocity
	ddtheta = joint acceleration

	Required parameters:
	robot - robot model from OpenRAVE based on jaco.urdf
	"""

	def __init__(self, robot):	
		""" Constructor, zeros out M,C,G values when created and

		Parameters:
		  robot     robot model from OpenRAVE
		"""
		self._robot = robot
		self.reset(robot)

	def reset(self, robot):
		""" Reset the state of this torque controller """
		self._pos_last = np.zeros((7,1)) # save last position for derivative 
		self._pos = np.zeros((7,1)) # current position
		self._vel = np.zeros((7,1)) # current velocity
		self._vel_last = np.zeros((7,1)) # save last velocity for acceleration
		self._accel = np.zeros((7,1)) # current acceleration
		self._cmd = np.zeros((7,1)) # command to send
		self._last_time = None # used for automatic calculation of dt

		self._pos = robot.GetDOFValues()
		self._vel = robot.GetDOFVelocities()

	@property
	def robot(self):
		"""Read-only access to the robot. """
		return self._robot

	@property
	def pos(self):
		""" Read-only access to the current position. """
		return self._pos

	@property
	def vel(self):
		""" Read-only access to the current velocity. """
		return self._vel

	@property
	def accel(self):
		""" Read-only access to the current acceleration. """
		return self._accel

	@property
	def last_time(self):
		""" Read-only access to the last time. """
		return self._last_time

	@property
	def cmd(self):
		""" Read-only access to the latest command. """
		return self._cmd

	def __str__(self):
		""" String representation of the current state of the controller. """
		result = ""
		result += "pos:  " + str(self.pos) + "\n"
		result += "vel:  " + str(self.vel) + "\n"
		result += "accel:  " + str(self.accel) + "\n"
		result += "cmd:     " + str(self.cmd) + "\n"
		return result

	def update_Tau(self, curr_pos, dt=None):
		"""  Update the torque control loop with nonuniform time step size.

		Parameters:
		curr_pos  current position of robot in joint space
		dt       Change in time since last call, in seconds, or None.
				If dt is None, then the system clock will be used to 
				calculate the time since the last update. 
		"""
		print "dt b: " + str(dt)
		if dt == None:
			curr_time = time.time()
			print "curr_t: " + str(curr_time)
			print "last_t: " + str(self._last_time)
			if self._last_time is None:
				self._last_time = curr_time 
			dt = curr_time - self._last_time
			print "dt inside: " + str(dt)
			self._last_time = curr_time
		
		print "dt a: " + str(dt)

		self._pos = curr_pos
		if dt == 0 or math.isnan(dt) or math.isinf(dt):
			return np.zeros((7,7)) # TODO or shold it be 0.0??

		# calculate velocity
		self._vel = (self._pos - self._pos_last) / dt
		self._pos_last = self._pos

		# calculate acceleration
		self._accel = (self._vel - self._vel_last) / dt
		self._vel_last = self._vel

		print "pos: " + str(self._pos)
		print "vel: " + str(self._vel)
		print "accel: " + str(self._accel)

		# set robot DOFs and Vels based on current state
		pos_vals = self._pos.reshape((7))
		pos_vals = np.append(pos_vals, [0, 0, 0])
		self.robot.SetDOFValues(pos_vals)

		vel_vals = self._vel.reshape((7))
		vel_vals = np.append(vel_vals, [0, 0, 0])
		self.robot.SetDOFVelocities(vel_vals)

		accel_vals = self._accel.reshape((7))
		accel_vals = np.append(accel_vals, [0,0,0])

		# compute torques
		tau = self.robot.ComputeInverseDynamics(accel_vals)

		# format output so you can send to ros topic
		self._cmd = tau[0:7].reshape((7,1))

		print "cmd: " + str(self._cmd)

		return self._cmd

if __name__ == "__main__":
	model_filename = 'jaco_dynamics'
	env, robot = initialize(model_filename)

	physics = RaveCreatePhysicsEngine(env,'ode')
	env.SetPhysicsEngine(physics)
	physics.SetGravity(np.array((0,0,0))) #should be (0,0,-9.8)

	env.StopSimulation()
	env.StartSimulation(timestep=0.001)

	robot.SetActiveDOFs(np.array([0, 1, 2, 3, 4, 5, 6]))
	viewer = env.GetViewer()
	viewer.SetCamera([[ 0.94684722, -0.12076704,  0.29815376,  0.21004671],
	[-0.3208323 , -0.42191214,  0.84797216, -1.40675116],
	[ 0.0233876 , -0.89855744, -0.43823231,  1.04685986],
	[ 0.        ,  0.        ,  0.        ,  1.        ]])

	controller = TorqueController(robot)
	print controller

	pos = np.array([180,180,180,180,180,180,180]).reshape((7,1))*(math.pi/180.0)
	controller.update_Tau(pos)
	print controller

	time.sleep(5)

	pos = np.array([190,180,180,180,180,180,180]).reshape((7,1))*(math.pi/180.0)
	controller.update_Tau(pos)
	print controller


