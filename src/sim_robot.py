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
import trajopt_planner

prefix = 'j2s7s300_driver'

pos1 = [14.30,162.95,190.75,124.03,176.10,188.25,167.94]
pos2 = [121.89,159.32,213.20,109.06,153.09,185.10,170.77]

class SimRobot(object):

	def __init__(self):	
	
		p1 = pos1 #candlestick_pos
		p2 = pos2 #home_pos
		start = np.array(p1)*(math.pi/180.0)
		goal = np.array(p2)*(math.pi/180.0)
		T = 6.0
		# create the trajopt planner from start to goal
		self.planner = trajopt_planner.Planner(start, goal, T)

		cartesian = self.planner.get_cartesian_waypts()
		self.planner.plot_cartesian_waypts(cartesian)
		#self.planner.execute_path_sim()

		rospy.init_node('sim_robot', anonymous=True)
	
		joint_angles_pub = rospy.Publisher(prefix+'/out/joint_angles', kinova_msgs.msg.JointAngles, queue_size=10)
		#joint_torques_pub = rospy.Publisher(prefix+'/out/joint_torques', kinova_msgs.msg.JointTorque, queue_size=10)
		joint_vel_pub = rospy.Publisher(prefix + '/out/joint_velocity', kinova_msgs.msg.JointVelocity, queue_size=10)	

		rospy.Subscriber(prefix+'/in/joint_velocity', kinova_msgs.msg.JointVelocity, self.joint_vel_callback, queue_size=1)

		rate = rospy.Rate(100) # 100hz

		while not rospy.is_shutdown():
			joint_angles = self.update_joint_angles()
			joint_vel = self.update_joint_velocities()

			joint_angles_pub.publish(joint_angles)
			joint_vel_pub.publish(joint_vel)

			rate.sleep()

	def update_joint_angles(self):
		joint_angles = kinova_msgs.msg.JointAngles()
		robot = self.planner.get_robot()
		joint_angles.joint1 = robot.GetDOFValues()[0]*(180.0/math.pi)
		joint_angles.joint2 = robot.GetDOFValues()[1]*(180.0/math.pi)
		joint_angles.joint3 = robot.GetDOFValues()[2]*(180.0/math.pi)
		joint_angles.joint4 = robot.GetDOFValues()[3]*(180.0/math.pi)
		joint_angles.joint5 = robot.GetDOFValues()[4]*(180.0/math.pi)
		joint_angles.joint6 = robot.GetDOFValues()[5]*(180.0/math.pi)
		joint_angles.joint7 = robot.GetDOFValues()[6]*(180.0/math.pi)

		#print "joint angles: " + str(robot.GetDOFValues())	
		return joint_angles

	def update_joint_velocities(self):
		joint_taus = kinova_msgs.msg.JointVelocity()
		robot = self.planner.get_robot()
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
		robot = self.planner.get_robot()
		robot.SetJointTorques(torque_curr,True)

	def joint_vel_callback(self, msg):

		curr_vel = np.array([msg.joint1,msg.joint2,msg.joint3,msg.joint4,msg.joint5,msg.joint6,msg.joint7, 0, 0, 0])

		# apply the torques to the sim robot
		robot = self.planner.get_robot()
		#curr_vel = [0,0,0,0,0,0,1,0,0,0]
		curr_vel = curr_vel*0.01 #scale down 
		robot.SetDOFVelocities(curr_vel)



if __name__ == '__main__':
	#model_filename = 'jaco_dynamics'
	#env, robot = initialize(model_filename)
	#env, robot = initialize()

	# enable the physics engine
	#physics = RaveCreatePhysicsEngine(env,'ode')
	#env.SetPhysicsEngine(physics)
	#physics.SetGravity(np.array((0,0,-9.8))) #should be (0,0,-9.8)

	#env.StopSimulation()
	#env.StartSimulation(timestep=0.001)

	#robot.SetActiveDOFs(np.array([0, 1, 2, 3, 4, 5, 6]))

	# DOF and velocity limits
	#n = robot.GetDOF()
	#dof_lim = robot.GetDOFLimits()
	#vel_lim = robot.GetDOFVelocityLimits()

	#print "dof_lim: " + str(dof_lim)
	#print "vel_lim: " + str(vel_lim)

	#robot.SetDOFLimits(-3*np.ones(n),3*np.ones(n))
	#robot.SetDOFVelocityLimits(100*vel_lim)

	# setup the viewer POV and the background
	#viewer = env.GetViewer()
	#viewer.SetCamera([[ 0.05117,  0.09089, -0.99455,  2.74898],
	#				   [ 0.99847,  0.01639,  0.05287, -0.17906],
	#				   [ 0.02111, -0.99573, -0.08991,  0.98455],
	#				   [ 0. , 0. , 0. , 1. ]])
	#viewer.SetBkgndColor([0.8,0.8,0.8])

	try:
		sim_robot = SimRobot()
	except rospy.ROSInterruptException:
		pass
