#!/usr/bin/env python3
import numpy as np

############################################################################
# This is a library that generates a trajectory based on a quintic polynomial
# It takes in multiple waypoints and generates a piecewise continuous
# polynomial between the waypoints
# Main variables
# List of waypoints - this should be an nx4 array (x, y, z, heading)
# coeffs - this is a (n-1) x 6 matrix, as there will be n-1 piecewise
# continuous polynomials to stitch the trajectory
# Current trajectory: for input with multiple waypoints, 
# a simple trajectory will be generated where intermediate waypoints
# will be given maximum speed, and acceleration and jerk between
# waypoints must be continuous
# There are a couple of overlapping functions in this currently
# 1. Mission planning - this takes in a list of waypoints and breaks it down into 
# a trajectory (acting more like a global planner and should be refactored out)
# 2. Trajectory generator - this is the actual quintic polynomial generator
# ############################################################################

class QuinticGenerator():

    def __init__(self) -> None:        
        self.waypoints = None
        self.coeffs = None
        self.max_speed = None
        self.max_accel = None

    # Waypoints should be a 3 x n trajectory
    # Where n is the number of waypoints to hit
    def set_waypoints(self, waypoints) -> None:
        if(waypoints.shape[0] != 3):
            raise Exception("Incorrect shape input in waypoints")

        if(waypoints.shape[1] < 2):
            raise Exception("Error: 2 waypoints needed at minimum")

        if(waypoints.shape[1] == 2):
            if(np.linalg.norm(waypoints[:][1] - waypoints[:][0], 2) < 0.5):
                raise Exception("Error: Trajectory is too short")

        self.waypoints = waypoints
        
    def set_max_speed(self, max_speed) -> None:
        self.max_speed = max_speed

    def set_max_accel(self, max_accel) -> None:
        self.max_accel = max_accel

    def get_full_state_waypoints(self) -> np.array:
        return self.full_state_waypoints

    # This creates a new mission 
    def new_mission(self, waypoints, max_speed, max_accel) -> np.array:
        self.set_waypoints(waypoints)
        self.set_max_speed(max_speed)
        self.set_max_accel(max_accel)
        full_state_waypoints = self.compute_full_state_waypoints(waypoints)
        self.full_state_waypoints = full_state_waypoints
        x_coeffs = np.zeros((waypoints.shape[1]-1, 6))
        y_coeffs = np.zeros((waypoints.shape[1]-1, 6))
        z_coeffs = np.zeros((waypoints.shape[1]-1, 6))
        t_segment = np.zeros((waypoints.shape[1]-1,1))
        for i in range(0, waypoints.shape[1]-1):
            x_0 = np.squeeze(full_state_waypoints[0,:,i])
            x_f = np.squeeze(full_state_waypoints[0,:,i+1])
            y_0 = np.squeeze(full_state_waypoints[1,:,i])
            y_f = np.squeeze(full_state_waypoints[1,:,i+1])
            z_0 = np.squeeze(full_state_waypoints[2,:,i])
            z_f = np.squeeze(full_state_waypoints[2,:,i+1])
            dist = np.linalg.norm(np.array([x_f[0] - x_0[0],
                                            y_f[0] - y_0[0],
                                            z_f[0] - z_0[0]]))
            delta_t = dist/self.max_speed
            
            t_segment[i] = delta_t
            x_c, y_c, z_c = self.compute_coeffs(x_0, x_f, y_0, y_f, z_0, z_f, 0, delta_t)
            x_coeffs[:][i] = x_c
            y_coeffs[:][i] = y_c
            z_coeffs[:][i] = z_c
        return x_coeffs, y_coeffs, z_coeffs, t_segment

    # This breaks the waypoints down into the fullstate
    # waypoints is a 3 x n matrix 
    # full state waypoints should be a 3 x 3 x n matrix
    # 1st dimension - x, y, z
    # 2nd dimension comes from the derivatives
    # 3rd dimension - waypoint segment
    # Current behavior: z should be 0 at start and end, 
    # All accelerations should be 0 at the waypoints
    # Assume 0 climb rate through the waypoints (z_d = 0) -> x_d ^2 + y_d^2 = v_max ^2
    def compute_full_state_waypoints(self, waypoints) -> np.array:
        n_waypoints = waypoints.shape[1]
        full_state_waypoints = np.zeros((waypoints.shape[0], 3, n_waypoints)) 

        for i in range(n_waypoints):

            # Set position waypoints
            full_state_waypoints[:, 0, i] = waypoints[:, i]

            # Set velocity values for intermediate waypoints
            if(i > 0 and i < (n_waypoints-1)):
                prev_heading = np.arctan2(waypoints[0][i]-waypoints[0][i-1], waypoints[1][i] - waypoints[1][i-1])
                next_heading = np.arctan2(waypoints[0][i+1]-waypoints[0][i], waypoints[1][i+1]-waypoints[1][i])
                heading = 0.5*(prev_heading + next_heading)
                full_state_waypoints[:, 1, i] = np.array([self.max_speed*np.sin(heading), self.max_speed*np.cos(heading), 0])

        return full_state_waypoints
        
    # Computes all of the piecewise polynomials for a fullstate
    # x, y, z are full states (position, velocity, acceleration)
    # on each axes of motion
    def compute_coeffs(self, x_0, x_f,
                            y_0, y_f,
                            z_0, z_f,
                            t_0, t_f) -> np.array:
        x_coeffs = self.compute_piecewise_coeffs_per_axes(x_0, x_f, t_0, t_f)
        y_coeffs = self.compute_piecewise_coeffs_per_axes(y_0, y_f, t_0, t_f)
        z_coeffs = self.compute_piecewise_coeffs_per_axes(z_0, z_f, t_0, t_f)
        return x_coeffs, y_coeffs, z_coeffs

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

    # coeffs is a 3x6 array containing the coefficients for interpolation
    # returns a 3x3 array, 
    # 1st dim are x, y, z
    # 2nd dim refers to waypoints
    def interpolate_trajectory(self, x_coeffs, y_coeffs, z_coeffs,
                            curr_t, t_start) -> np.ndarray:
        x = np.poly1d(np.squeeze(x_coeffs))
        x_d = np.polyder(x, 1)
        x_dd = np.polyder(x, 2)

        y = np.poly1d(np.squeeze(y_coeffs))
        y_d = np.polyder(y, 1)
        y_dd = np.polyder(y, 2)

        z = np.poly1d(np.squeeze(z_coeffs))
        z_d = np.polyder(z, 1)
        z_dd = np.polyder(z, 2)

        delta_t = curr_t - t_start
        x_interp = np.array([np.polyval(x, delta_t), 
                    np.polyval(x_d, delta_t),
                    np.polyval(x_dd, delta_t)])
        y_interp = np.array([np.polyval(y, delta_t), 
            np.polyval(y_d, delta_t),
            np.polyval(y_dd, delta_t)])
        z_interp = np.array([np.polyval(z, delta_t), 
            np.polyval(z_d, delta_t),
            np.polyval(z_dd, delta_t)])
        return x_interp, y_interp, z_interp


        
