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

def plotTable(env):
	"""
	Plots the robot table in OpenRAVE.
	"""
	# load table into environment
	objects_path = find_in_workspaces(
			project='interactpy',
			path='envdata',
			first_match_only=True)[0]
	env.Load('{:s}/table.xml'.format(objects_path))
	table = env.GetKinBody('table')
	table.SetTransform(np.array([[0.0, 1.0,  0.0, 0],
  								 [1.0, 0.0,  0.0, 0],
			                     [0.0, 0.0,  1.0, -0.832],
			                     [0.0, 0.0,  0.0, 1.0]]))
	color = np.array([0.9, 0.75, 0.75])
	table.GetLinks()[0].GetGeometries()[0].SetDiffuseColor(color)

def plotTableMount(env,bodies):
	"""
	Plots the robot table mount in OpenRAVE.
	"""
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

def plotLaptop(env,bodies):
	"""
	Plots the robot table mount in OpenRAVE.
	"""
	# create robot base attachment
	body = RaveCreateKinBody(env, '')
	#12 x 9 x 1 in, 0.3048 x 0.2286 x 0.0254 m
	# divide by 2: 0.1524 x 0.1143 x 0.0127
	#20 in from robot base
	body.InitFromBoxes(np.array([[0,0,0,0.1143,0.1524,0.0127]]))
	body.SetTransform(np.array([[1.0, 0.0,  0.0, -0.508],
			                     [0.0, 1.0,  0.0, 0],
			                     [0.0, 0.0,  1.0, -0.032],
			                     [0.0, 0.0,  0.0, 1.0]]))
	body.SetName("laptop")
	env.Add(body, True)
	color = np.array([0, 0, 0])
	body.GetLinks()[0].GetGeometries()[0].SetDiffuseColor(color)
	bodies.append(body)

env, robot = interact.initialize_empty('jaco_dynamics', envXML='environ.env.xml', empty=True)
bodies = []

viewer = env.GetViewer()
viewer.SetSize(1000,1000)
#viewer.SetCamera([
#	[0.,  0., -1., 0.462949],
#	[1.,  0.,  0., 2.192697],
#	[0., -1.,  1., 0.936772],
#	[0.,  0.,  0., 1.]])
#viewer.SetBkgndColor([0.8,0.8,0.8])


plotTable(env)
plotTableMount(env,bodies)
plotLaptop(env,bodies)

cartesianDOFs = robotToCartesian(robot)
#for coord in cartesianDOFs:
#	plotCartesian(env, bodies, coord)
print "cartesian dofs: " + str(cartesianDOFs)
time.sleep(20)


