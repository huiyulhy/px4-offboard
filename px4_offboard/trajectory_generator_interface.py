#!/usr/bin/env python3
############################################################################
# This is a ROS2 interface class which takes in the position once
# and uses the quintic polynomial library to generate a trajectory
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

class TrajectoryGeneratorInterface(Node):

    def __init__(self):
        super().__init__('quintic_generator')
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
            durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            depth=1
        )

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
        self.local_position_ned = None
        self.start_time = None

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
        if(not self.trajectory_generated):
            self.trajectory_generated = True
            self.start_time = self.local_position_ned.timestamp/1.0e6
        print("Self local position")
        print(self.local_position_ned.x)
        print(self.local_position_ned.y)
        print(self.local_position_ned.z)
        print(self.start_time)

    # Command loop callback
    def cmdloop_callback(self):
        # Publish offboard control modes
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = int(Clock().now().nanoseconds / 1000)
        offboard_msg.position=True
        offboard_msg.velocity=False
        offboard_msg.acceleration=False
        self.publisher_offboard_mode.publish(offboard_msg)
        if (self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD and self.arming_state == VehicleStatus.ARMING_STATE_ARMED):

            trajectory_msg = TrajectorySetpoint()
            trajectory_msg.position[0] = self.radius * np.cos(self.theta)
            trajectory_msg.position[1] = self.radius * np.sin(self.theta)
            trajectory_msg.position[2] = -2.0
            self.publisher_trajectory.publish(trajectory_msg)

            self.theta = self.theta + self.omega * self.dt

def main(args=None):
    rclpy.init(args=args)

    trajectory_generator_interface = TrajectoryGeneratorInterface()

    rclpy.spin(trajectory_generator_interface)

    trajectory_generator_interface.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
             