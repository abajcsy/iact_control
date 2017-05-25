import numpy as np
from openravepy import *
import collections
from interactpy import TrajoptPlanner
import TrajectoryUtils
import interactpy

from jaco import Jaco

def dofToTransform(robot, dof):
    if len(dof) == 7:
        dof = np.append(dof, np.array([1,1,1]))
    with robot:
        #print dof
        robot.SetDOFValues(dof)
        return robot.GetActiveManipulator().GetTransform()

def dofToCartesian(robot, dof):
    return transformToCartesian(dofToTransform(robot, dof))

def transformToCartesian(transform):
    return transform[0:3, 3]

def plotPoint(env, bodies, coords):
    with env:
        body = RaveCreateKinBody(env, '')
        body.InitFromBoxes(np.array([[coords[0], coords[1], coords[2],
                                      0.01, 0.01, 0.01]]))
        body.SetName(str(len(bodies)))
        env.Add(body, True)
        bodies.append(body)

def plotSpheres(env, coords):
    handles = []
    handles.append(env.plot3(points=coords, pointsize=0.02,  colors=(0,0,1,0.2),
                             drawstyle=1))
    return handles

def plotPoints(env, bodies, coords, color=None):
    if color == None:
        color = np.array([0, 1, 0]) #green by default
    for x, y, z in coords:
        body = RaveCreateKinBody(env, '')
        body.InitFromBoxes(np.array([[x, y, z, 0.01, 0.01, 0.01]]))
        body.SetName(str(len(bodies)))
        env.Add(body, True)
        body.GetLinks()[0].GetGeometries()[0].SetDiffuseColor(color)
        bodies.append(body)


def plotNewPoint(env, robot, trajectory, index, deltas, bodies):
    dof = trajectory.GetAllWaypoints2D()[index]
    transform = dofToTransform(robot, dof)
    newTransform = move(transform, deltas[0], deltas[1], deltas[2])
    coords = transformToCartesian(newTransform)

    body = RaveCreateKinBody(env, '')
    body.InitFromBoxes(np.array([[coords[0], coords[1], coords[2],
                                  0.02, 0.02, 0.02]]))
    body.SetName(str(len(bodies)))
    env.Add(body, True)
    bodies.append(body)

def removePoints(env, bodies):
    for body in bodies:
        env.Remove(body)

def plotWaypoints(env, robot, waypoints):

    bodies = []
    for waypoint in waypoints:
        dof = np.append(waypoint, np.array([1, 1, 1]))
        coord = transformToCartesian(dofToTransform(robot, dof))
        plotPoint(env, bodies, coord)

    return bodies

def waypointsToCartesian(robot, waypoints):
    if isinstance(waypoints[0], collections.Iterable):
        coords = []
        for waypoint in waypoints:
            coords.append(transformToCartesian(dofToTransform(robot, waypoint)))
        return np.array(coords)
    return transformToCartesian(dofToTransform(robot, waypoints))


def move(transform, dx, dy, dz):
    
        trans = np.array([[1, 0, 0, dx],
                          [0, 1, 0, dy],
                          [0, 0, 1, dz],
                          [0, 0, 0, 1]])
        return np.dot(trans, transform)


def moveRobot(env, robot, dof, dx, dy, dz):

    
    #tp = TrajoptPlanner.TrajoptPlanner(env)

    if len(dof) == 7:
        dof = np.append(dof, np.array([1,1,1]))
    with robot:
        robot.SetDOFValues(dof)
        transform = robot.GetActiveManipulator().GetEndEffectorTransform()
        new_transform = move(transform, dx, dy, dz)

        print("original", transform)
        print("new", new_transform)

        #path = tp.PlanToIK(robot, new_transform)
        #return path.GetAllWaypoints2D()[-1]
        archie = Jaco(robot)
        return archie.FindIKSolution(new_transform)
        
def modifyTrajectory(env, robot, trajectory, index, deltas):

    dof = trajectory.GetAllWaypoints2D()[index]
    print "original dof", dof
    newDof = moveRobot(env, robot, dof, deltas[0], deltas[1], deltas[2])
    print "new dof", newDof

    return TrajectoryUtils.AddChangeToTrajectory(trajectory, index, newDof, env,
                                                 keep_times=False)

