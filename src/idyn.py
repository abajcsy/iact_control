import interactpy
import numpy as np
from openravepy import *
from interactpy import initialize
from interactpy import demo

dofvalues = np.array([-1, 2, 0, 2, 0, 4, 0, 1.11022302e-16,  -1.11022302e-16, 3.33066907e-16])
dofvelocities = np.array([0.1,0.1,0.1,0.1,0.2,0.2,0.2,0,0,0])
dofaccel = np.array([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0,0,0])

def compute_idyn(robot):
	"""
	Returns torque computed from equation of motion.
	
	tau = M(dofvalues) * dofaccel + C(dofvalues,dofvel) * dofvel + G(dofvalues)

	"""

	print "dofvel: " + str(dofvelocities)
	print "dofaccel: " + str(dofaccel)

	robot.SetDOFValues(dofvalues)
	robot.SetDOFVelocities(dofvelocities)

	# returns tau_interia, tau_coriolis, tau_gravity: 
	# tau_M, tau_C, tau_G = robot.ComputeInverseDynamics(dofaccel, None, returncomponents=True)
	tau = robot.ComputeInverseDynamics(dofaccel)

	#print "tau_M: " + str(tau_M)
	#print "tau_C: " + str(tau_C)
	#print "tau_G: " + str(tau_G)
	#print "tau: " + str(tau_M + tau_C + tau_G)
	print "tau: " + str(tau)

if __name__ == '__main__':
	# setup robot and environment
	env, robot = initialize()
	robot.SetActiveDOFs(np.array([0, 1, 2, 3, 4, 5, 6]))
	viewer = env.GetViewer()
	viewer.SetCamera([[ 0.94684722, -0.12076704,  0.29815376,  0.21004671],
		   [-0.3208323 , -0.42191214,  0.84797216, -1.40675116],
		   [ 0.0233876 , -0.89855744, -0.43823231,  1.04685986],
		   [ 0.        ,  0.        ,  0.        ,  1.        ]])

	# test default demo
	#demo(robot)

	# compute inverse dynamics given robot's current acceleration
	compute_idyn(robot)


