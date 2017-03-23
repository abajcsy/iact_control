import numpy as np
import math
from sympy import symbols, lambdify, Point, Line, Segment
from numpy import linspace
import matplotlib.pyplot as plt

T = 1.0
t_i = 0.5
theta_i = 165*math.pi/180

iact = np.array([165,197.13,180.070,43.4309,265.11,257.271,287.9276]).reshape((7,1))* (math.pi/180.0)

s = np.array([180]*7).reshape((7,1))* (math.pi/180.0)
g = np.array([103.366,197.13,180.070,43.4309,265.11,257.271,287.9276]).reshape((7,1))* (math.pi/180.0)

def proj_interact_pt(t_i, theta_i, s, g):
	p1,p2 = Point(0,s), Point(T,g)
	l1 = Line(p1,p2)
	p3 = Point(t_i,theta_i)
	return l1.projection(p3)

def proj_interact(t_i, iact, s, g):
	t = np.array([0.0]*7).reshape((7,1))
	projected = np.array([0.0]*7).reshape((7,1))
	p = None
	for i in range(7):
		p = proj_interact_pt(t_i, iact[i][0], s[i][0], g[i][0])
		projected[i][0] = p.y
		t[i][0] = p.x
	return (t, projected)

if __name__ == '__main__':
	t = symbols('t')
	theta = (g-s)*(1/T)*t+s
	print theta

	plt.figure(1)
	for i in range(7):
		l = "j"+str(i)

		lam_theta = lambdify(t, theta[i][0], modules=['numpy'])

		x_vals = np.linspace(0,T,500)
		y_vals = lam_theta(x_vals)

		plt.plot(x_vals, y_vals, label=l)

	# compute projection of interaction point onto line
	(t_new, proj) = proj_interact(t_i, iact, s, g)

	plt.plot(t_new[2],proj[2],marker='o', color='m')
	plt.plot(0.5,165*math.pi/180,marker='o', color='r')
	plt.xlabel('time (s)')
	plt.ylabel('theta')
	plt.legend()

	"""
	Only for joint 1 
	"""

	# after interaction
	plt.figure(2)
	theta_func = lambdify(t, theta[0][0], modules=['numpy'])
	theta_expected = theta_func(t_i)

	# compute projection of interaction point onto line
#	p3_proj = proj_interact_pt(t_i, theta_i, s[0], g[0])

#	print p3_proj.x - t_i

	x_vals = np.linspace(0,T,500)
	y_vals = theta_func(x_vals)

	plt.plot(x_vals, y_vals, label="j=0")

	plt.plot(t_i,theta_i,marker='o', color='r')
	plt.plot(t_i,theta_expected,marker='o', color='b')
#	plt.plot(p3_proj.x,p3_proj.y,marker='o', color='m')

	plt.xlabel('time (s)')
	plt.ylabel('theta')
	plt.legend()
	plt.show()

