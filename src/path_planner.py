import numpy as np
import math
from sympy import symbols
from numpy import linspace
from sympy import lambdify
import matplotlib.pyplot as plt
import time

def linear_path(start,goal,T):
	"""
	Returns linear (in C-space) time-parametrized labda function for each joint
	"""
	t = symbols('t')
	theta = (goal-start)*(1/T)*t + start

	lin_func = [None]*7
	for i in range(7):
		lam_theta = lambdify(t, theta[i][0], modules=['numpy'])
		lin_func[i] = lam_theta

	return lin_func

if __name__ == '__main__':
	T = 20.0
	s = np.array([180]*7).reshape((7,1))* (math.pi/180.0)
	g = np.array([103.366,197.13,180.070,43.4309,265.11,257.271,287.9276]).reshape((7,1))* (math.pi/180.0)

	lin = linear_path(s,g,T)
	print lin[0](19.9999)
