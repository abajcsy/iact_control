import numpy as np
import math
from sympy import symbols
from numpy import linspace
from sympy import lambdify
import matplotlib.pyplot as plt

T = 20.0

s = np.array([180]*7).reshape((7,1))* (math.pi/180.0)
g = np.array([103.366,197.13,180.070,43.4309,265.11,257.271,287.9276]).reshape((7,1))* (math.pi/180.0)

t = symbols('t')
theta = (g-s)*(1/T)*t+s

for i in range(7):
	l = "j"+str(i)

	lam_theta = lambdify(t, theta[i][0], modules=['numpy'])

	x_vals = np.linspace(0,20,500)
	y_vals = lam_theta(x_vals)

	plt.plot(x_vals, y_vals, label=l)

plt.xlabel('time (s)')
plt.ylabel('theta')
plt.legend()
plt.show()

