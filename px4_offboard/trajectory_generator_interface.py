#!/usr/bin/env python3
############################################################################
# This is a ROS2 interface class which takes in the position once
# and uses the quintic polynomial library to generate a trajectory
# This interface is using NED coordinates to generate the trajectory
# ############################################################################
import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import VehicleStatus
from px4_msgs.msg import VehicleLocalPosition

from quintic_generator import QuinticGenerator

class TrajectoryGeneratorInterface(Node):

    def __init__(self):
        super().__init__('quintic_generator')
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
            durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            depth=1
        )

        self.quintic_generator = QuinticGenerator()

        self.status_sub = self.create_subscription(
            VehicleStatus,
            '/fmu/out/vehicle_status',
            self.vehicle_status_callback,
            qos_profile)

        self.local_position_ned_sub = self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position',
            self.vehicle_local_position_ned_callback,
            qos_profile
        )

        self.publisher_offboard_mode = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.publisher_trajectory = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        timer_period = 0.02 #seconds
        self.timer = self.create_timer(timer_period, self.cmdloop_callback)
        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX
        self.arming_state = VehicleStatus.ARMING_STATE_DISARMED
        self.trajectory_generated = False
        self.trajectory_started = False
        self.local_position_ned = None
        self.starting_local_position_ned = None
        self.trajectory_start_time = None
        self.x_coeffs = np.zeros((6, 1))
        self.y_coeffs = np.zeros((6, 1))
        self.z_coeffs = np.zeros((6, 1))
        self.t_seg = 0

    # Vehicle status callback
    def vehicle_status_callback(self, msg):
        # TODO: handle NED->ENU transformation
        print("NAV_STATUS: ", msg.nav_state)
        print("  - offboard status: ", VehicleStatus.NAVIGATION_STATE_OFFBOARD)
        self.nav_state = msg.nav_state
        self.arming_state = msg.arming_state

    # Current local position callback
    def vehicle_local_position_ned_callback(self, msg):
        # Do we need to handle vehicle local position callback here?
        self.local_position_ned = msg
        if(self.local_position_ned.xy_valid):
            if(not self.trajectory_generated):
                self.trajectory_generated = True
                self.starting_local_position_ned = msg
                waypoints = np.transpose(np.array([[0.0, 0.0, 0.0],
                        [5.0, 0.0, -3.0]]))
                self.x_coeffs, self.y_coeffs, self.z_coeffs, self.t_seg = self.quintic_generator.new_mission(waypoints, 0.3, 0.3)
                print(self.x_coeffs)
                print(self.t_seg)
    # Command loop callback
    def cmdloop_callback(self):
        # Publish offboard control modes
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = int(Clock().now().nanoseconds / 1000)
        offboard_msg.position=True
        offboard_msg.velocity=True
        offboard_msg.acceleration=True
        self.publisher_offboard_mode.publish(offboard_msg)
        if (self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD and self.arming_state == VehicleStatus.ARMING_STATE_ARMED):
            trajectory_msg = TrajectorySetpoint()
            if(self.trajectory_started == False):
                self.trajectory_started = True
                self.trajectory_start_time = int(Clock().now().nanoseconds/1.0e9)

                x_i, y_i, z_i = self.quintic_generator.interpolate_trajectory(self.x_coeffs, self.y_coeffs, self.z_coeffs, 0, 0)
                print(x_i)
                trajectory_msg.position[0] = x_i[0] + self.starting_local_position_ned.x
                trajectory_msg.velocity[0] = x_i[1]
                trajectory_msg.acceleration[0] = x_i[2]

                trajectory_msg.position[1] = y_i[0] + self.starting_local_position_ned.y
                trajectory_msg.velocity[1] = y_i[1]
                trajectory_msg.acceleration[1] = y_i[2]

                trajectory_msg.position[2] = z_i[0] + self.starting_local_position_ned.z
                trajectory_msg.velocity[2] = z_i[1]
                trajectory_msg.acceleration[2] = z_i[2]
            else:
                curr_time = int(Clock().now().nanoseconds/1.0e9)
                # if trajectory ended, hover
                if(curr_time - self.trajectory_start_time > self.t_seg):
                    print(curr_time - self.trajectory_start_time)
                    trajectory_msg.position[0] = self.local_position_ned.x
                    trajectory_msg.position[1] = self.local_position_ned.y
                    trajectory_msg.position[2] = self.local_position_ned.z
                else:
                    x_i, y_i, z_i = self.quintic_generator.interpolate_trajectory(self.x_coeffs, self.y_coeffs, self.z_coeffs, curr_time, self.trajectory_start_time)
                    trajectory_msg.position[0] = x_i[0] + self.starting_local_position_ned.x
                    trajectory_msg.velocity[0] = x_i[1]
                    trajectory_msg.acceleration[0] = x_i[2]

                    trajectory_msg.position[1] = y_i[0] + self.starting_local_position_ned.y
                    trajectory_msg.velocity[1] = y_i[1]
                    trajectory_msg.acceleration[1] = y_i[2]

                    trajectory_msg.position[2] = z_i[0] + self.starting_local_position_ned.z
                    trajectory_msg.velocity[2] = z_i[1]
                    trajectory_msg.acceleration[2] = z_i[2]                
            self.publisher_trajectory.publish(trajectory_msg)


def main(args=None):
    rclpy.init(args=args)

    trajectory_generator_interface = TrajectoryGeneratorInterface()

    rclpy.spin(trajectory_generator_interface)

    trajectory_generator_interface.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
             