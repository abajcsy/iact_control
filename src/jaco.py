import numpy as np
import openravepy as rave
from openravepy.misc import InitOpenRAVELogging
from openravepy import interfaces
InitOpenRAVELogging()

"""
env = rave.Environment()
env.SetViewer('qtcoin')
module = rave.RaveCreateModule(env, 'urdf')
name = module.SendCommand('load /home/ubuntu/catkin_ws/src/jaco.urdf /home/ubuntu/catkin_ws/src/jaco.srdf')
r = env.GetRobot(name)
r.SetDOFValues([0,3,0,2,0,4,0,0,0,0])
r.SetActiveManipulator("j2s7s300")
"""

# The Jaco class wraps a generic openrave robot for
# easy typical use in the InterACT Lab at Berkeley
class Jaco:
    # robot
    # ik6d
    # manipinterface

    def __init__(self, robot):
        self.robot = robot
        self.ik6d = rave.databases.inversekinematics.InverseKinematicsModel(
            robot,
            iktype=rave.IkParameterization.Type.Transform6D
        )
        if not self.ik6d.load():
            self.ik6d.autogenerate()
        self.manipinterface = interfaces.BaseManipulation(robot)

    # FinkIkSolution delegates to the underlying 6d ik solver,
    # Use for direct access to iksolutions, if want to set the
    # jaco to a particular transform, use SetTransform
    def FindIKSolution(self, transform):
        return self.ik6d.manip.FindIKSolution(transform, rave.IkFilterOptions.CheckEnvCollisions)

    # GetTransform retrieves the transform of the principal manipulator
    def GetTransform(self):
        return self.robot.GetActiveManipulator().GetTransform()

    # SetTransform solves the inverse kinematics and then puts the jaco
    # into the configuration of the solution. Throws an exception if
    # no solution is found
    def SetTransform(self, transform):
        sol = self.FindIKSolution(transform)
        if sol is None:
            raise Exception("no ik solution")
        self.robot.SetDOFValues(sol, self.ik6d.manip.GetArmIndices())

    # MoveToConfiguration uses the BRRT planner in openrave to
    # move from the current config to the goal config
    def MoveToConfiguration(self, q):
        if len(q) > 7:
            q = q[0:7]

        self.manipinterface.MoveManipulator(q)

    # MoveToTransform moves the jaco to the new transform
    def MoveToTransform(self, t):
        self.MoveToConfiguration(self.FindIKSolution(t))

    # Configuration retrieves the configuration vector in C_space for the Jaco
    # It will be 10-dimension, with the first 7 DOFs corresponding to the arm and
    # the final three to the fingers.
    def Configuration(self):
        return self.robot.GetDOFValues()

    # SetConfiguration puts the jaco in a given configuration.
    # Use to set the entire configuration vector, if just chaning
    # one DOF, try the individual methods. i.e., SetFinger1
    def SetConfiguration(self, vals):
        return self.robot.SetDOFValues(vals)

    # SetFingers helps for setting the three finger DOFs
    def SetFingers(self, one, two, three):
        c = self.Configuration()
        c[-3] = one
        c[-2] = two
        c[-1] = three
        self.SetConfiguration(c)

    # SetFinger1 helps for setting the first (bottom) finger
    def SetFinger1(self, val):
        c = self.Configuration()
        c[-3] = val
        self.SetConfiguration(c)

    # SetFinger2 helps for setting the second (top-left) finger
    def SetFinger2(self, val):
        c = self.Configuration()
        c[-2] = val
        self.SetConfiguration(c)

    # SetFinger3 helps for setting the third (top-right) finger
    def SetFinger3(self, val):
        c = self.Configuration()
        c[-1] = val
        self.SetConfiguration(c)

#archie = Jaco(r)

#print("Welcome to InterACT. The JACO is the symbol `archie`")
