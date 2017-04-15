#! /usr/bin/env python
import math
import numpy as np

l_vals = [0,1] 		# 0 = slow, 1 = fast
tau_vals = [0,1]	# 0 = low_torque, 1 = high_torque
tauH_vals = [0,1]	# 0 = sense_slow, 1 = sense_fast

belief_l = [0.3, 0.7] # belief initially over l=slow, and l=fast

# Defines the probability of P(l=slow) and P(l=fast)
def l_pmf(l):
	if l == 0:
		return 0.5
	elif l == 1:
		return 0.5

# Defines the probability of P(l_t | l_{t-1})
def l_conditioned(l_curr, l_prev):
	if l_curr == l_prev:
		return 1.0
	else:
		return 0.0

# Defines the probability of P(tauH_t | l_t)
def tauH_conditioned(tauH, l_curr):
	if tauH == 1 and l_curr == 1:
		return 0.7
	elif tauH == 0 and l_curr == 1:
		return 0.3
	elif tauH == 1 and l_curr == 0:
		return 0.2
	elif tauH == 0 and l_curr == 0:
		return 0.8

# Runs bayes filter given the current belief distribution over l, 
# and a given tau command and tauH measurement
def bayes_filter(bel_l, tau, tauH):
	bel_bar = [None]*len(l_vals)
	for l_curr in l_vals:
		bel_bar[l_curr] = l_conditioned(l_curr, 0)*bel_l[l_curr] + l_conditioned(l_curr, 1)*bel_l[l_curr]	
		bel_l[l_curr] = tauH_conditioned(tauH, l_curr)*bel_bar[l_curr] 
	
	# normalize
	n = bel_l[0] + bel_l[1]
	bel_l[0] /= n
	bel_l[1] /= n
	return bel_l

if __name__ == "__main__":
	print "original belief over l: " + str(belief_l)

	tau = 1 # high torque
	tauH = 0 # sense_slow

	for i in range(20):
		tauH = int(i < 7)
		#print "tau: " + str(tau) + ", tauH: " + str(tauH)
		belief_l = bayes_filter(belief_l, tau, tauH)
		print "updated belief over l: " + str(belief_l)

