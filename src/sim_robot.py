#!/usr/bin/env python
# license removed for brevity
import rospy
import roslib; roslib.load_manifest('kinova_demo')
from kinova_msgs.srv import *
from std_msgs.msg import Float32
import kinova_msgs.msg
import numpy as np
import math
import time

from openravepy import *
import interactpy
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
		# read the current joint torques from the robot
		torque_curr = np.array([msg.joint1,msg.joint2,msg.joint3,msg.joint4,msg.joint5,msg.joint6,msg.joint7,0,0,0])	

		print "joint tau: " + str(torque_curr)
		self.robot.SetJointTorques(torque_curr,True)
			
		# save running list of joint torques
		#self.joint_torques = np.column_stack((self.joint_torques,torque_curr))
		
		#for i in range(7):
		#	if np.fabs(torque_curr[i][0]) > 7:
		#		print "I HAVE SET THE INTERACTION"
		#		self.interaction = True
		#		break
			#else: 
				#self.interaction = False

		# update the plot of joint torques over time
		#t = time.time() - self.process_start_T
		#self.plotter.update_joint_torque(torque_curr, t)


if __name__ == '__main__':
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

	try:
		sim_robot = SimRobot(robot)
	except rospy.ROSInterruptException:
		pass
