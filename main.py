from concurrent.futures import ProcessPoolExecutor
from concurrent.futures._base import Future as FutureType
from copy import deepcopy
from math import sqrt
from os import system
from pickle import dump
from random import randint, choice, random
from sys import exit
from gcodeparser import GcodeParser
from jax import value_and_grad
from jax.experimental.optimizers import adam
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt
plt.ioff()
from matplotlib.animation import FuncAnimation
from tqdm import tqdm_notebook as tqdm

import warnings
warnings.filterwarnings("ignore", message="No GPU/TPU found, falling back to CPU.")

ProcessPool = ProcessPoolExecutor()

#user config
filename = input("Please enter the filename of the gcode you wish to analyse    ")
#this is whether you want the exturder to never leave the print
#the end result will likely be very similar to the starting state if this is enabled
snap_to_print = True if y in input("Would you like to enable snap to print? y/n    ") else False
try:
    target_x = float(input("Enter the target x point or leave this blank for no target    "))
except ValueError:
    #-1 is used to represent no target point
    target_x = -1.0
try:
    target_y = float(input("Enter the target y point or leave this blank for no target    "))
except ValueError:
    target_y = -1.0
                     
#the square root in normal euclidean distance means we don't get a derivaitve in terms of x so this version is used in the loss
def squared_distance(point_1, point_2):
    x_distance = (point_1[0] - point_2[0])**2
    y_distance = (point_1[1] - point_2[1])**2
    return x_distance + y_distance

def distance(point_1, point_2):
    '''
    Takes the euclidean distance between two points
    :param point_1: list, takes the form [x, y] where x and y are coordinates
    :param point_2: list, takes the form [x, y] where x and y are coordinates
    :returns: float, the euclidean distance between the two points
    '''
    return sqrt(squared_distance(point_1, point_2))

def loss(snapshot_points, target_points, valid_points):
    '''
    What we are trying to minimize
    Consitsts of the distance between each snapshot coordinate and the previous snapshot coordinate,
    the distance between each snapshot coridnate and the nearest one within the print
    and the distance between each snapshot cooridage and the target coordinate
    :param snapshot_points: list, contains each point where a snapshot will be taken, this is what is being learnt
    :param valid_points: list, the nearest point, within the print, to each entry in snapshot_points,
    :returns: float, the loss
    '''
    prev_snapshot_point = (-1, -1)
    prev_valid_point = (-1, -1)
    loss = 0
    num = 0
    for snapshot_point, target_point, valid_point in zip(snapshot_points, target_points, valid_points):
        #-1 is used if there is no previous point
        if -1 not in prev_snapshot_point:
            loss += squared_distance(snapshot_point, prev_snapshot_point)
            num += 2

        #if no -1's are anywhere we can just use the distance formula
        if -1 not in target_point:
            loss += squared_distance(snapshot_point, target_point)
            num += 2 
        #else we only get the distance for the axis on which there isn't a -1
        elif target_point[0] != -1:
            loss += (snapshot_point[0] - target_point[0])**2
            num += 1
        elif target_point[1] != -1:
            loss += (snapshot_point[0] - target_point[0])**2
            num += 1
        
        loss += squared_distance(snapshot_point, valid_point)
        num += 2

        prev_snapshot_point = snapshot_point
        prev_valid_point = valid_point
    #takes the mean
    return loss / num

def nearest_point(point, points):
    '''Finds the point in points that is nearest to point'''

    #based on https://github.com/OpenGenus/cosmos/blob/master/code/divide_conquer/src/closest_pair_of_points/closest_pair.py
    def calc_dist(point_1, point_2):
        x_distance = (point_1[0] - point_2[0])**2
        y_distance = (point_1[1] - point_2[1])**2
        return sqrt((x_distance + y_distance))
    if len(points) > 2:
        return nearest_point(point, [
            nearest_point(point, points[:len(points) // 2]),
            nearest_point(point, points[:len(points) // 2])
        ])
    else:
        min_dist = 1000
        best_point = (-1, -1)
        if type(points[0]) != list and type(points[0]) != FutureType:
            return points
        for possible_point in points:
            if type(possible_point) == FutureType:
                possible_point = possible_point.result()
            dist = calc_dist(point, possible_point)
            if dist < min_dist:
                min_dist = dist
                best_point = possible_point
        return best_point

def get_nearest_points(points, target_points):
    '''
    Runs nearest_point on each point in points, assings each call of nearest_point a CPU core
    '''
    return ProcessPool.map(nearest_point, points, target_points)


gradient = value_and_grad(loss)

#gets the number of layers in the gcode file along with every point that extrusion and movement is occuring at
num_layers, valid_points = GcodeParser(filename)

#exact values for targets can help to break ties between getting near to the print and getting near to other snapshot points
target_points = [[target_x, target_y] for i in range(0, num_layers)]

nearest_points = list(get_nearest_points(target_points, valid_points))
#inits snapshot points with the points within the print that are nearest to the target points
snapshot_points = deepcopy(nearest_points)
target_loss = loss(snapshot_points, target_points, nearest_points)

#creates an animated plot of this inital state
def update(n):
    snapshot_plot.set_data(snapshot_points[n][0], snapshot_points[n][1])
    nearest_plot.set_data(nearest_points[n][0], nearest_points[n][1])

fig = plt.figure()
ax = plt.axes(xlim=(0, 150), ylim=(0, 150))
snapshot_plot, = ax.plot([], [], "ro", markersize=5)
nearest_plot, = ax.plot([], [], "go", markersize=5)
animation = FuncAnimation(fig, update, frames=len(snapshot_points), interval=15)
animation.save("start.gif", writer="imagemagick")

print("")

#inits our optimizer
init, update, get_value = adam(1)
adam_state = init(snapshot_points)

epoch = 0
progress_bar = tqdm()
num_stops = 0
prev_loss = 0
got_latest_points = False
try:
    while True:
        #gets the new points back from the gradient decent if we haven't already
        if not got_latest_points:
            snapshot_points = get_value(adam_state)
            #generators are awesome but they mess with JAX's autodifferentation so we convert to a list
            nearest_points = list(get_nearest_points(snapshot_points, valid_points))
            got_latest_points = True
        #gets the gradients and the loss
        curr_loss, curr_grads = gradient(snapshot_points, target_points, nearest_points)
        i = 0

        adam_state = update(i, curr_grads, adam_state)
        got_latest_points = False
        #we peridocly print the loss
        if epoch % 10 == 0:
            print(curr_loss)
        #if we have reached the minimum (local or global)
        if abs(prev_loss - curr_loss) < 0.00001:
            #quit
            raise KeyboardInterrupt

        epoch += 1
        progress_bar.update(1)
        prev_loss = curr_loss

except:
    if not got_latest_points:
        snapshot_points = get_value(adam_state)
        nearest_points = list(get_nearest_points(snapshot_points, valid_points))
        got_latest_points = True
    #the processpool gets broken when we keyboard interrupt so it has to be recreated
    ProcessPool = ProcessPoolExecutor()
    
    if snap_to_print:
        #replaces every snapshot point with the one nearest to it in the print
        snapshot_points = list(get_nearest_points(snapshot_points, valid_points))
                     
    print(loss(snapshot_points, target_points, nearest_points))

    #creates an anmated plot of the end state
    def update(n):
        snapshot_plot.set_data(snapshot_points[n][0], snapshot_points[n][1])
        nearest_plot.set_data(nearest_points[n][0], nearest_points[n][1])

    fig = plt.figure()
    ax = plt.axes(xlim=(0, 150), ylim=(0, 150))
    snapshot_plot, = ax.plot([], [], "ro", markersize=5)
    nearest_plot, = ax.plot([], [], "go", markersize=5)
    animation = FuncAnimation(fig, update, frames=len(snapshot_points), interval=15)
    animation.save("end.gif", writer="imagemagick")
    with open('points.pickle', 'wb') as fp:
        dump(snapshot_points, fp)
    exit()
