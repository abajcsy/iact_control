#! /usr/bin/env python
import rospy
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import rospy
import math
import kinova_msgs.msg
from kinova_msgs.msg import KinovaPose
import time

class RVizualizer(object):
	"""
	This class visualizes trajectories, poses, robots in RVIZ.
	"""

	def __init__(self):
		rospy.init_node('rvizualizer')

		marker_topic = "/visualization_marker_array"
		pose_topic = "/j2s7s300_driver/out/cartesian_command"
		
		self.marker_pub = rospy.Publisher(marker_topic, MarkerArray, queue_size=1)
		self.marker_array = MarkerArray()
		self.counter = 0

		# create subscriber to end-effector pose
		rospy.Subscriber(pose_topic, kinova_msgs.msg.KinovaPose, self.viz_trajectory, queue_size=1)

		while not rospy.is_shutdown():
			# Publish the MarkerArray
			self.marker_pub.publish(self.marker_array)

			rospy.sleep(0.01)

	def viz_trajectory(self, msg):
		print self.counter
		marker = Marker()
		marker.id = self.counter
		marker.header.frame_id = "/root"
		marker.header.seq = self.counter
		marker.header.stamp.secs = self.counter
		marker.type = marker.SPHERE
		marker.action = marker.ADD
		marker.scale.x = 0.02
		marker.scale.y = 0.02
		marker.scale.z = 0.02
		marker.color.a = 1.0
		marker.color.r = 1.0
		marker.color.g = 0.0
		marker.color.b = 0.0
		marker.pose.orientation.w = 1.0
		marker.pose.position.x = msg.X
		marker.pose.position.y = msg.Y 
		marker.pose.position.z = msg.Z

		self.marker_array.markers.append(marker)
		self.counter += 1

	def viz_waypts(self, waypts):
		

if __name__ == '__main__':
	viz = RVizualizer()

