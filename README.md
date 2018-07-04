# iact_control: InterACTive Learning & Control

Control, planning, and learning system for physical human-robot interaction (pHRI) with a JACO2 7DOF robotic arm. 

## Dependencies
* Ubuntu 14.04, ROS Indigo, OpenRAVE, Python 2.7
* or_trajopt, or_urdf, or_rviz, prpy, pr_ordata
* kinova-ros
* fcl

## Running the System
### Setting up the JACO2 Robot
Turn the robot on and put it in home position by pressing and holding the center (yellow) button on the joystick.
 
In a new terminal, turn on the Kinova API by typing:
```
roslaunch kinova_bringup kinova_robot.launch kinova_robotType:=j2s7s300 use_urdf:=true
```
### Starting the controller, planner, and learning system
In another terminal, run:
```
roslaunch iact_control trajoptPID.launch ID:=0 task:=0 methodType:=A demo:=F record:=F
```
Command-line options include:
* `ID`: Participant/user identification number (for experiments and data saving)
* `task`: Task number {Distance to human = 0, Cup orientation = 1, Distance to table = 2, Distance to laptop = 3}
* `methodType`: Sets the pHRI control method {impedance control = A, impedance + learning from pHRI = B}
* `demo`: Demonstrates the "optimal" way to perform the task {default = F, optimal demo = T}
* `record`: Records the interaction forces, measured trajectories, and cost function weights for a task {record data = T, don't record = F}

### Publications
Code used in the following papers:
* A. Bajcsy* , D.P. Losey*, M.K. O'Malley, and A.D. Dragan. [Learning Robot Objectives from Physical Human Robot Interaction.](http://proceedings.mlr.press/v78/bajcsy17a/bajcsy17a.pdf) Conference on Robot Learning (CoRL), 2017.
* A. Bajcsy , D.P. Losey, M.K. O'Malley, and A.D. Dragan. [Learning from Physical Human Corrections, One Feature at a Time.](https://dl.acm.org/citation.cfm?id=3171267) International Confernece on Human-Robot Interaction (HRI), 2018.


### References
* TrajOpt Planner: http://rll.berkeley.edu/trajopt/doc/sphinx_build/html/index.html
* PID Control Reference: https://w3.cs.jmu.edu/spragunr/CS354/handouts/pid.pdf
