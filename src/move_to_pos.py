#! /usr/bin/env python
"""A test program to test action servers for the JACO and MICO arms."""

import roslib; roslib.load_manifest('kinova_demo')

import actionlib

import kinova_msgs.msg
import geometry_msgs.msg
import tf
import std_msgs.msg
import math
import thread
from kinova_msgs.srv import *
import argparse
from robot_control_modules import *

prefix = 'j2s7s300_'
nbJoints = 7
interactive = True
duration_sec = 100


if __name__ == '__main__':
	if len(sys.argv) < 9:
		print "ERROR: Not enough arguments. Specify p_gains, i_gains, d_gains."
	else:	
		pose = [float(sys.argv[2]),float(sys.argv[3]),float(sys.argv[4]),float(sys.argv[5]),float(sys.argv[6]),float(sys.argv[7]),float(sys.argv[8])]
		print "Moving to pose: " + str(pose)

	try:        
		#prefix, nbJoints = argumentParser(None)	
		rospy.init_node('torque_compensated_mode')
		if (interactive == True):        
			nb = raw_input("Moving robot to position, press return to start, n to skip")
		if (nb != "n" and nb != "N"):

			result = joint_position_client(pose, prefix)

		print("Done!")
	except rospy.ROSInterruptException:
		print "program interrupted before completion"
