#!/usr/bin/env python3
import numpy as np

############################################################################
# This is a library that generates a trajectory based on a quintic polynomial
# It takes in multiple waypoints and generates a piecewise continuous
# polynomial between the waypoints
# Main variables
# List of waypoints - this should be an nx3 array
# coeffs - this is a (n-1) x 6 matrix, as there will be n-1 piecewise
# continuous polynomials to stitch the trajectory
# Current trajectory: for input with multiple waypoints, 
# a simple trajectory will be generated where intermediate waypoints
# will be given maximum speed, and acceleration and jerk between
# waypoints must be continuous
# ############################################################################

class QuinticGenerator():

    def __init__(self) -> None:        
        self.waypoints = None
        self.coeffs = None
        self.max_speed = None

    def set_waypoints(self, waypoints):
        self.waypoints = waypoints
        
    def set_max_speed(self, max_speed):
        self.max_speed = max_speed

    # Computes all of the piecewise polynomials for a fullstate
    #def compute_coeffs(self, full_states)

    # Computes the coefficient between 2 points
    # x1 and x2 are full state values up to acceleration
    # this is for a single axes
    # we will always normalize to take start as t_0
    def compute_piecewise_coeffs_per_axes(self, x_0, x_f, t_0, t_f):
        t_f = t_f - t_0
        t_0 = 0
        mat = np.array([[pow(t_0, 5.0), pow(t_0, 4), pow(t_0, 3), pow(t_0, 2), t_0, 1],
                        [pow(t_f, 5.0), pow(t_f, 4), pow(t_f, 3), pow(t_f, 2), t_f, 1],
                        [5.0*pow(t_0, 4), 4.0*pow(t_0, 3), 3.0*pow(t_0, 2), 2.0*t_0, 1, 0],
                        [5.0*pow(t_f, 4), 4.0*pow(t_f, 3), 3.0*pow(t_f, 2), 2.0*t_f, 1, 0],
                        [20.0*pow(t_0, 3), 12.0*pow(t_0, 2), 6.0*t_0, 2, 0, 0],
                        [20.0*pow(t_f, 3), 12.0*pow(t_f, 2), 6.0*t_f, 2, 0, 0]])
        rhs = np.array([x_0[0], x_f[0], x_0[1], x_f[1], x_0[2], x_f[2]])
        coeffs = np.linalg.solve(mat, rhs)
        return coeffs