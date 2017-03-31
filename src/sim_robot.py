#!/usr/bin/env python
# license removed for brevity
import rospy
import roslib; roslib.load_manifest('kinova_demo')

import kinova_msgs.msg
import numpy as np
import math
import time
import interactpy

from kinova_msgs.srv import *
from std_msgs.msg import Float32
from openravepy import *
from interactpy import initialize

prefix = 'j2s7s300_driver'

class SimRobot(object):

	def __init__(self, robot):	
		self._robot = robot
	
		rospy.init_node('sim_robot', anonymous=True)
	
		joint_angles_pub = rospy.Publisher(prefix+'/out/joint_angles', kinova_msgs.msg.JointAngles, queue_size=10)
		joint_torques_pub = rospy.Publisher(prefix+'/out/joint_torques', kinova_msgs.msg.JointTorque, queue_size=10)
	
		rospy.Subscriber(prefix + '/in/joint_torque', kinova_msgs.msg.JointTorque, self.joint_torques_callback, queue_size=1)

		rate = rospy.Rate(100) # 100hz

		while not rospy.is_shutdown():
			joint_angles = self.update_joint_angles()
			joint_torques = self.update_joint_torques()

			joint_angles_pub.publish(joint_angles)
			joint_torques_pub.publish(joint_torques)

			rate.sleep()

	@property
	def robot(self):
		"""Read-only access to the robot. """
		return self._robot	

	def update_joint_angles(self):
		joint_angles = kinova_msgs.msg.JointAngles()
		joint_angles.joint1 = self.robot.GetDOFValues()[0]*(180.0/math.pi)
		joint_angles.joint2 = self.robot.GetDOFValues()[1]*(180.0/math.pi)
		joint_angles.joint3 = self.robot.GetDOFValues()[2]*(180.0/math.pi)
		joint_angles.joint4 = self.robot.GetDOFValues()[3]*(180.0/math.pi)
		joint_angles.joint5 = self.robot.GetDOFValues()[4]*(180.0/math.pi)
		joint_angles.joint6 = self.robot.GetDOFValues()[5]*(180.0/math.pi)
		joint_angles.joint7 = self.robot.GetDOFValues()[6]*(180.0/math.pi)

		#print "joint angles: " + str(self.robot.GetDOFValues())	
		return joint_angles

	def update_joint_torques(self):
		# TODO THIS IS WRONG AND IS PUBLISHING VELOCITIES
		joint_taus = kinova_msgs.msg.JointTorque()
		joint_taus.joint1 = robot.GetDOFVelocities()[0]
		joint_taus.joint2 = robot.GetDOFVelocities()[1]
		joint_taus.joint3 = robot.GetDOFVelocities()[2]
		joint_taus.joint4 = robot.GetDOFVelocities()[3]
		joint_taus.joint5 = robot.GetDOFVelocities()[4]
		joint_taus.joint6 = robot.GetDOFVelocities()[5]
		joint_taus.joint7 = robot.GetDOFVelocities()[6]
		return joint_taus

	def joint_torques_callback(self,msg):
		"""
		Reads the latest torque sensed by the robot and records it for 
		plotting & analysis
		"""
		# read the joint torques sent from the controller
		torque_curr = np.array([msg.joint1,msg.joint2,msg.joint3,msg.joint4,msg.joint5,msg.joint6,msg.joint7,0,0,0])	

		# apply the torques to the sim robot
		self.robot.SetJointTorques(torque_curr,True)


if __name__ == '__main__':
	model_filename = 'jaco_dynamics'
	env, robot = initialize(model_filename)

	# enable the physics engine
	physics = RaveCreatePhysicsEngine(env,'ode')
	env.SetPhysicsEngine(physics)
	physics.SetGravity(np.array((0,0,-9.8))) #should be (0,0,-9.8)

	env.StopSimulation()
	env.StartSimulation(timestep=0.001)

	robot.SetActiveDOFs(np.array([0, 1, 2, 3, 4, 5, 6]))

	# DOF and velocity limits
	n = robot.GetDOF()
	dof_lim = robot.GetDOFLimits()
	vel_lim = robot.GetDOFVelocityLimits()

	print "dof_lim: " + str(dof_lim)
	print "vel_lim: " + str(vel_lim)

	#robot.SetDOFLimits(-3*np.ones(n),3*np.ones(n))
	#robot.SetDOFVelocityLimits(100*vel_lim)

	# setup the viewer POV and the background
	viewer = env.GetViewer()
	viewer.SetCamera([[ 0.05117,  0.09089, -0.99455,  2.74898],
					   [ 0.99847,  0.01639,  0.05287, -0.17906],
					   [ 0.02111, -0.99573, -0.08991,  0.98455],
					   [ 0. , 0. , 0. , 1. ]])
	viewer.SetBkgndColor([0.2,0.2,0.2])
	#env.ShowWorldAxes(1)

	try:
		sim_robot = SimRobot(robot)
	except rospy.ROSInterruptException:
		pass
