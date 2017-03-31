#! /usr/bin/env python
import rospy
import math

import numpy as np
import matplotlib.pyplot as plt

num_joints = 7

class Plotter(object):	
	"""
	This class implements basic plotting functionality for PID control. 
	
	
	Required parameters:
		p_gain, i_gain, d_gain    - gain terms for the PID controller
	"""
	def __init__(self,p,i,d):
		self.p_gain = p
		self.i_gain = i
		self.d_gain = d

		self.times = np.zeros(1)
		self.p_error = np.zeros((7,1))
		self.i_error = np.zeros((7,1))
		self.d_error = np.zeros((7,1))

		# for tracking applied command torque
		self.cmd = np.zeros((7,1))

		# for tracking measured torque and force direction
		self.joint_torques = np.zeros((7,1))
		self.joint_times = np.zeros(1)
		# force direction and magnitude
		self.force_dir = np.zeros(1)
		self.force_mag = np.zeros(1)
		
		self.joint_vels = np.zeros((7,1))
		self.vels_times = np.zeros(1)

		self.path_start_time = 0

	def set_path_start_time(self,t):
		self.path_start_time = t

	def update_joint_torque(self, j_torque, force_d, force_m, t):
		"""
		Updates joint torque measurements based on most recent movement.
		"""
		self.joint_torques = np.column_stack((self.joint_torques,j_torque))
		self.joint_times = np.column_stack((self.joint_times,np.array(t)))

		self.force_dir = np.column_stack((self.force_dir,np.array(force_d)))
		self.force_mag = np.column_stack((self.force_mag,np.array(force_m)))

	def update_joint_vel(self, j_vel, t):
		"""
		Updates joint velocity measurements based on most recent movement.
		"""
		self.joint_vels = np.column_stack((self.joint_vels,j_vel))
		self.vels_times = np.column_stack((self.vels_times,np.array(t)))
	
	def update_PID_plot(self, p_e, i_e, d_e, cmd, t):
		"""
		Updates the P,I,D errors based on most recent movement.
		"""
		self.p_error = np.column_stack((self.p_error,p_e))
		self.i_error = np.column_stack((self.i_error,i_e))
		self.d_error = np.column_stack((self.d_error,d_e))
		self.cmd = np.column_stack((self.cmd,cmd))
		self.times = np.column_stack((self.times,np.array(t)))

	def plot_tau_PID(self, total_path_time):
		"""
		Plots the commanded and measured torques over time.
		"""
		c = ['b','g','r','c','m','y','#FF8C00']

		for i in range(num_joints):
			ax = plt.subplot(7, 1, i+1)
			l = "j"+str(i)
			t = self.joint_times[0]
			torques = self.joint_torques[i]	

			print len(t)
			print len(self.force_dir)

			# plot the joint torque over time 
			base_line,  = plt.plot(t, torques, '-', linewidth=3.0, color=c[i], label=str(l)+" measured_tau")
			# self.times[0]
			plt.plot(self.times[0], self.cmd[i], '-', linewidth=3.0, color='k', label=str(l)+" cmd_tau")
			# plot force direction
			plt.plot(t, self.force_dir[0], '--', linewidth=1.5, color='#808080', label=str(l)+" force dir")
			# plot force magnitude
			plt.plot(t, self.force_mag[0], '-', linewidth=1.5, color='#808080', label=str(l)+" force mag")

			plt.axvline(self.path_start_time, color='#808080')
			if i == 0:
				plt.title("Measured Torque and P,I,D Torque Command")
			plt.ylabel("torque (Nm)")
			plt.legend(prop={'size':10})
			plt.grid()

			axes = plt.gca()
			axes.set_xlim([3,15])
			axes.set_ylim([-10,10])


		# plot torque commands over time
		"""
		plt.subplot(2, 1, 2)
		for i in range(num_joints):	
			l = "j"+str(i)
			plt.plot(self.times[0], self.cmd[i], '-', linewidth=3.0, label=l)
		plt.xlabel("time (s)")
		plt.axvline(self.path_start_time, color='#808080')
		plt.ylabel("cmd torque (Nm)")
		plt.legend(prop={'size':10})
		plt.grid()

		axes = plt.gca()
		axes.set_xlim([3,10])
		axes.set_ylim([-12,12])
		"""

		plt.show()

	def plot_PID(self, total_path_time):
		"""
		Plots the P,I,D errors over time.
		"""

		"""
		# check that all dimensions match up. If not, correct
		min_len = len(self.times[0])
		min_o = self.times
		need_correction = False
		objs = [self.times, self.i_error, self.p_error, self.d_error, self.joint_vels, self.joint_torques]
		for o in objs:
			if len(o[0]) < min_len:
				min_len = len(o[0])
				min_o = o
				need_correction = True
		
		if need_correction:
			for o in objs:
				if o is not min_o:
					diff = np.abs(len(o[0]) - min_len)
					o = o[:,0:diff]
		"""
	
		# plot p_error
		ax = plt.subplot(6, 1, 1)
		for i in range(num_joints):	
			l = "j"+str(i)
			plt.plot(self.times[0], self.p_error[i], '-', linewidth=3.0, label=l)
		plt.axvline(self.path_start_time, color='#808080')
		ax.text(0.95, 0.01, 'Path time: '+str(total_path_time)+" s", verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes, color='black', fontsize=12)
		plt.ylabel("p error (rad)")
		plt.title("P,I,D error with K_p: " + str(self.p_gain) + ", K_i:" + str(self.i_gain) + ", K_d:" + str(self.d_gain))
		plt.legend(prop={'size':10})
		plt.grid()

		# plot i_error
		plt.subplot(6, 1, 2)
		for i in range(num_joints):	
			l = "j"+str(i)
			plt.plot(self.times[0], self.i_error[i], '-', linewidth=3.0, label=l)
		plt.axvline(self.path_start_time, color='#808080')
		plt.ylabel("i error")
		plt.legend(prop={'size':10})
		plt.grid()

		# plot d_error
		plt.subplot(6, 1, 3)
		for i in range(num_joints):	
			l = "j"+str(i)
			plt.plot(self.times[0], self.d_error[i], '-', linewidth=3.0, label=l)
		plt.axvline(self.path_start_time, color='#808080')
		plt.ylabel("d error")
		plt.legend(prop={'size':10})
		plt.grid()

		
		# plot joint_torques
		ax = plt.subplot(6, 1, 4)
		for i in range(num_joints):	
			l = "j"+str(i)
			t = self.joint_times[0]
			torques = self.joint_torques[i]			
			avg = np.average(torques)
			stdev_high = [avg + np.std(torques)]*len(t)
			stdev_low = [avg - np.std(torques)]*len(t)

			# plot the joint torque over time 
			base_line,  = plt.plot(t, torques, '-', linewidth=3.0, label=l)

			# plot average of each joint torque over time
			plt.plot(t, [avg]*len(t), '--', linewidth=1.0, color=base_line.get_color())

			# plot +/- 1 standard deviation around mean
			ax.fill_between(t, stdev_low, stdev_high, facecolor=base_line.get_color(), alpha=0.2)
		plt.axvline(self.path_start_time, color='#808080')
		plt.ylabel("measuresd torque (Nm)")
		plt.legend(prop={'size':10})
		plt.grid()

		# plot torque commands over time
		plt.subplot(6, 1, 5)
		for i in range(num_joints):	
			l = "j"+str(i)
			plt.plot(self.times[0], self.cmd[i], '-', linewidth=3.0, label=l)
		plt.xlabel("time (s)")
		plt.axvline(self.path_start_time, color='#808080')
		plt.ylabel("cmd torque (Nm)")
		plt.legend(prop={'size':10})
		plt.grid()

		# plot difference between command torque and measured torque
		plt.subplot(6, 1, 6)
		for i in range(num_joints):	
			l = "j"+str(i)
			plt.plot(self.times[0], self.cmd[i]-self.joint_torques[i], '-', linewidth=3.0, label=l)
		plt.xlabel("time (s)")
		plt.axvline(self.path_start_time, color='#808080')
		plt.ylabel("cmd - measured tau")
		plt.legend(prop={'size':10})
		plt.grid()

		# plot velocity measured
		"""
		plt.subplot(5, 1, 5)
		for i in range(num_joints):	
			l = "j"+str(i)
			plt.plot(self.vels_times[0], self.joint_vels[i], '-', linewidth=3.0, label=l)
		plt.xlabel("time (s)")
		plt.axvline(self.path_start_time, color='#808080')
		plt.ylabel("measured vel (rad/s)")
		plt.legend(prop={'size':10})
		plt.grid()
		"""

		plt.show()
		
	

