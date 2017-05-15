#! /usr/bin/env python
import rospy
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import geometry_msgs.msg
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
		marker_topic2 = "visualization_marker_array_waypts"
		pose_topic = "/j2s7s300_driver/out/cartesian_command"
		
		# marker publisher for real-time pose
		self.marker_pub = rospy.Publisher(marker_topic, MarkerArray, queue_size=1)
		self.marker_array = MarkerArray()
		self.counter = 0

		# marker publisher for waypoint data
		self.waypt_marker_pub = rospy.Publisher(marker_topic2, MarkerArray, queue_size=1)
		self.waypt_marker_array = MarkerArray()
		self.waypt_counter = 0

		self.waypt_pub = rospy.Publisher('/pose_arr', geometry_msgs.msg.PoseArray, queue_size=1)

		# create subscriber to real-time end-effector pose
		rospy.Subscriber(pose_topic, kinova_msgs.msg.KinovaPose, self.viz_trajectory, queue_size=1)
		# create subscriber to cartesian waypoints
		rospy.Subscriber('/cartesian_waypts', geometry_msgs.msg.PoseArray, self.viz_waypts, queue_size=1)
		

		while not rospy.is_shutdown():
			# Publish the MarkerArrays
			self.marker_pub.publish(self.marker_array)
			self.waypt_marker_pub.publish(self.waypt_marker_array)
			rospy.sleep(0.01)

	def viz_trajectory(self, msg):
		#print self.counter
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

	def viz_waypts(self, msg):
		for i in range(len(msg.poses)):
			pt = msg.poses[i].position
			marker = Marker()
			marker.id = self.waypt_counter
			marker.header.frame_id = "/root"
			marker.header.seq = self.waypt_counter
			marker.header.stamp.secs = self.waypt_counter
			marker.type = marker.SPHERE
			marker.action = marker.ADD
			marker.scale.x = 0.04
			marker.scale.y = 0.04
			marker.scale.z = 0.04
			marker.color.a = 1.0
			marker.color.r = 0.0
			marker.color.g = 0.0
			marker.color.b = 1.0
			marker.pose.orientation.w = 1.0
			marker.pose.position.x = pt.x
			marker.pose.position.y = pt.y
			marker.pose.position.z = pt.z

			self.waypt_marker_array.markers.append(marker)
			self.waypt_counter += 1

if __name__ == '__main__':
	viz = RVizualizer()

