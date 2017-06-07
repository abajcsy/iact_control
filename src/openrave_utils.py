import openravepy
from openravepy import *
from prpy.bind import bind_subclass
from archierobot import ArchieRobot
from catkin.find_in_workspaces import find_in_workspaces
import numpy as np
import logging
import math

# Silence the planning logger to prevent spam.
logging.getLogger('prpy.planning.base').addHandler(logging.NullHandler())

robot_starting_dofs = np.array([-1, 2, 0, 2, 0, 4, 0, 1.11022302e-16,  -1.11022302e-16, 3.33066907e-16])

def initialize(model_filename='jaco', envXML=None):
	'''
	Load and configure the JACO robot. If envXML is not None, loads environment.
	Returns robot and environment.
	'''
	env = openravepy.Environment()
	if envXML is not None:
		env.LoadURI(envXML)
	env.SetViewer('qtcoin')

	# Assumes the robot files are located in the data folder of the
	# kinova_description package in the catkin workspace.
	urdf_uri = 'package://iact_control/src/data/'+model_filename+'.urdf'
	srdf_uri = 'package://iact_control/src/data/'+model_filename+'.srdf'
	or_urdf = openravepy.RaveCreateModule(env, 'urdf')
	robot_name = or_urdf.SendCommand('load {:s} {:s}'.format(urdf_uri, srdf_uri))
	robot = env.GetRobot(robot_name)
	bind_subclass(robot, ArchieRobot)

	robot.SetActiveDOFs(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
	robot.SetDOFValues(robot_starting_dofs)

	viewer = env.GetViewer()
	viewer.SetSize(1000,1000)
	cam_params = np.array([[-0.99885711, -0.01248719, -0.0461361 , -0.18887213],
		   [ 0.02495645,  0.68697757, -0.72624996,  2.04733515],
		   [ 0.04076329, -0.72657133, -0.68588079,  1.67818344],
		   [ 0.        ,  0.        ,  0.        ,  1.        ]])
	viewer.SetCamera(cam_params)
	viewer.SetBkgndColor([0.8,0.8,0.8])

	return env, robot

# ------- Plotting & Conversion Utils ------- #

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

def plotTraj(env,robot,bodies,waypts, color=[0, 1, 0]):
	"""
	Plots the best trajectory found or planned
	"""
	for waypoint in waypts:
		dof = np.append(waypoint, np.array([1, 1, 1]))
		dof[2] += math.pi
		robot.SetDOFValues(dof)
		coord = robotToCartesian(robot)
		plotPoint(env, bodies, coord[6], 0.005, color)

def plotPoint(env, bodies, coords, size=0.1, color=[0, 1, 0]):
	"""
	Plots a single cube point in OpenRAVE at coords(x,y,z) location
	"""
	with env:
		c = np.array(color)
		body = RaveCreateKinBody(env, '')
		body.InitFromBoxes(np.array([[coords[0], coords[1], coords[2],
					  size, size, size]]))
		body.SetName("pt"+str(len(bodies)))
		env.Add(body, True)
		body.GetLinks()[0].GetGeometries()[0].SetDiffuseColor(c)
		bodies.append(body)

def plotTable(env):
	"""
	Plots the robot table in OpenRAVE.
	"""
	# load table into environment
	objects_path = find_in_workspaces(
			project='iact_control',
			path='src/data',
			first_match_only=True)[0]
	env.Load('{:s}/table.xml'.format(objects_path))
	table = env.GetKinBody('table')
	table.SetTransform(np.array([[0.0, 1.0,  0.0, -0.8128/2], #should be negative 1?
  								 [1.0, 0.0,  0.0, 0],
			                     [0.0, 0.0,  1.0, -0.1143], #-0.7874
			                     [0.0, 0.0,  0.0, 1.0]]))
	color = np.array([0.9, 0.75, 0.75])
	table.GetLinks()[0].GetGeometries()[0].SetDiffuseColor(color)

def plotMan(env):
	"""
	Plots a human in OpenRAVE.
	"""
	# load table into environment
	objects_path = find_in_workspaces(
			project='iact_control',
			path='src/data',
			first_match_only=True)[0]
	env.Load('{:s}/manifest.xml'.format(objects_path))


def plotTableMount(env,bodies):
	"""
	Plots the robot table mount in OpenRAVE.
	"""
	# create robot base attachment
	body = RaveCreateKinBody(env, '')
	body.InitFromBoxes(np.array([[0,0,0, 0.3048/2,0.8128/2,0.1016/2]]))
	body.SetTransform(np.array([[1.0, 0.0,  0.0, 0],
			                     [0.0, 1.0,  0.0, 0],
			                     [0.0, 0.0,  1.0, -0.1016/2],
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
	body.SetTransform(np.array([[1.0, 0.0,  0.0, -1.3858/2],
			                     [0.0, 1.0,  0.0, 0],
			                     [0.0, 0.0,  1.0, -0.1016],
			                     [0.0, 0.0,  0.0, 1.0]]))
	body.SetName("laptop")
	env.Add(body, True)
	color = np.array([0, 0, 0])
	body.GetLinks()[0].GetGeometries()[0].SetDiffuseColor(color)
	bodies.append(body)

def executePathSim(env,robot,waypts):
	"""
	Executes in the planned trajectory in simulation
	"""
	traj = RaveCreateTrajectory(env,'')
	traj.Init(robot.GetActiveConfigurationSpecification())
	for i in range(len(waypts)):
		traj.Insert(i, waypts[i])
	robot.ExecutePath(traj)
	
