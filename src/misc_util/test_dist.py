import interactpy
from interactpy import *
import numpy as np
import openravepy
from openravepy import *
import time

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

env, robot = interact.initialize_empty('jaco_dynamics', empty=True)
env.Load('{:s}/table.xml'.format(objects_path))
table = env.GetKinBody('table')
table.SetTransform(np.array([[1.00000000e+00, -2.79931237e-36,  1.41282351e-14, 1.50902510e-01],
                             [-2.07944729e-28,  1.00000000e+00,  1.47183799e-14, -1.47385532e-02],
                             [-1.41282351e-14, -1.47183799e-14,  1.00000000e+00, -1.00134850e-01],
                             [0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]]))

bodies = []
cartesianDOFs = robotToCartesian(robot)
for coord in cartesianDOFs:
	plotCartesian(env, bodies, coord)
print "cartesian dofs: " + str(cartesianDOFs)
time.sleep(20)


