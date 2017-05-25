import numpy as np
import sys, select, os

P = 15*np.eye(7)
D = 5*np.eye(7)

while True:
	if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
		line = raw_input()
		splt = line.split()

		if len(splt) == 3:
			joint_num = int(splt[1])
			gain = float(splt[2])
			if splt[0] == 'd':
				# modifying d gains			
				D[joint_num][joint_num] = gain
			elif splt[0] == "p":
				# modifying p gains
				P[joint_num][joint_num] = gain
		print "P: " + str(P)
		print "D: " + str(D)
		# if pressed enter, then quit
		if line == "":
			break

	
print "Done"
