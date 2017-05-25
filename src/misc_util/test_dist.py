import interactpy
from interactpy import *
import numpy as np
import openravepy
from openravepy import *
import time

from catkin.find_in_workspaces import find_in_workspaces

def plotCartesian(env,bodies,coords):
	"""
	Plots a single cube point in OpenRAVE at coords(x,y,z) location
	"""
	color = np.array([0, 1, 0])
	size = 0.06
	body = RaveCreateKinBody(env, '')
	body.InitFromBoxes(np.array([[coords[0], coords[1], coords[2],
				  size, size, size]]))
	body.SetName(str(len(bodies)))
	env.Add(body, True)
	body.GetLinks()[0].GetGeometries()[0].SetDiffuseColor(color)
	bodies.append(body)

def robotToCartesian(robot):
	"""
	Converts robot configuration into a list of cartesian 
	(x,y,z) coordinates for each of the robot's links.
	------
	Returns: 7-dimensional list of 3 xyz values
	"""
	links = robot.GetLinks()
	cartesian = [None]*7
	i = 0
	for i in range(1,8):
		link = links[i] 
		tf = link.GetTransform()
		cartesian[i-1] = tf[0:3,3]

	return cartesian

def create_table(env):
	# load table into environment
	objects_path = find_in_workspaces(
		    project='interactpy',
		    path='envdata',
		    first_match_only=True)[0]
	env.Load('{:s}/table.xml'.format(objects_path))
	table = env.GetKinBody('table')
	table.SetTransform(np.array([[1.0, 0.0,  0.0, 0],
		                         [0.0, 1.0,  0.0, 0],
		                         [0.0, 0.0,  1.0, -0.832],
		                         [0.0, 0.0,  0.0, 1.0]]))
	color = np.array([0.9, 0.75, 0.75])
	table.GetLinks()[0].GetGeometries()[0].SetDiffuseColor(color)

def create_table_mount(env, bodies):
	# create robot base attachment
	body = RaveCreateKinBody(env, '')
	body.InitFromBoxes(np.array([[0,0,0, 0.14605,0.4001,0.03175]]))
	body.SetTransform(np.array([[1.0, 0.0,  0.0, 0],
		                         [0.0, 1.0,  0.0, 0],
		                         [0.0, 0.0,  1.0, -0.032],
		                         [0.0, 0.0,  0.0, 1.0]]))
	body.SetName("robot_mount")
	env.Add(body, True)
	color = np.array([0.9, 0.58, 0])
	body.GetLinks()[0].GetGeometries()[0].SetDiffuseColor(color)
	bodies.append(body)

env, robot = interact.initialize_empty('jaco_dynamics', empty=True)
bodies = []

create_table(env)
create_table_mount(env,bodies)

cartesianDOFs = robotToCartesian(robot)
#for coord in cartesianDOFs:
#	plotCartesian(env, bodies, coord)
print "cartesian dofs: " + str(cartesianDOFs)
time.sleep(20)


