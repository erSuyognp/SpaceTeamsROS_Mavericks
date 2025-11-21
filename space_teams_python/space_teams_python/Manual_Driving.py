#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Quaternion
from space_teams_definitions.srv import Float, String
import numpy as np
import math
from space_teams_python.transformations import *

class LoggingRover(Node):
    def __init__(self):
        super().__init__('LoggingRover')

        # ---- Service clients ----
        self.logger_client = self.create_client(String, 'log_message')
        self.core_sampling_client = self.create_client(Float, 'CoreSample')
        self.brake_client = self.create_client(Float, 'Brake')
        self.steer_client = self.create_client(Float, 'Steer')
        self.accel_client = self.create_client(Float, 'Accelerator')

        # ---- Subscribers ----
        self.current_location = None
        self.current_velocity = None
        self.current_rotation = None

        self.create_subscription(Point, 'LocationLocalFrame', self.location_cb, 10)
        self.create_subscription(Point, 'VelocityLocalFrame', self.velocity_cb, 10)
        self.create_subscription(Quaternion, 'RotationLocalFrame', self.rotation_cb, 10)

        # ---- Waypoints (your exact 20 sampling points) ----
        self.waypoints = [
            np.array([-54.31019727, 191.84449903, -19.54598818]),
            np.array([111.24089259, 427.56166121, -54.81398767]),
            np.array([-349.10709106, 558.01869306, -68.71836618]),
            np.array([1281.36380015, 1647.50529027, -39.35361376]),
            np.array([654.62948546, 1186.61595725, -48.4778713]),
            np.array([-606.74433428, 332.44253661, -20.41775233]),
            np.array([1349.86835614, 1047.23075279, -46.89420337]),
            np.array([231.41034119, -858.69285702, -63.3150879]),
            np.array([45.56236659, 921.05755228, -65.76412603]),
            np.array([1960.32237043, 1423.88737415, -89.97019481]),
            np.array([1098.14343253, 1987.40560248, -45.45757708]),
            np.array([10.15805303, -752.47151722, -68.15878792]),
            np.array([1532.81368707, 1255.13690297, -48.47378546]),
            np.array([-561.74721182, 28.52558036, -29.92751284]),
            np.array([1958.28017108, 1381.24222162, -76.75680176]),
            np.array([-1025.65838348, 274.39353778, -76.31593519]),
            np.array([410.36797363, -956.93367913, -84.31272572]),
            np.array([247.67056987, 579.07900331, -75.04176954]),
            np.array([345.53461945, 1330.35839896, -73.5301525]),
            np.array([1073.3882324, 1613.84763245, -50.72357905])
        ]

        self.waypoint_order = [0, 1, 17, 8, 2, 5, 15, 13, 11, 7, 16, 18, 4, 19, 10, 3, 6, 12, 14, 9]


        self.current_wp_idx = 0
        self.tolerance = 5.0  # meters

        self.timer = self.create_timer(0.1, self.timer_callback)
        self.get_logger().info("Logging Rover (with core sampling) started.")

    # --------- Callbacks ----------
    def location_cb(self, msg):
        self.current_location = np.array([msg.x, msg.y, msg.z])

    def velocity_cb(self, msg):
        self.current_velocity = np.array([msg.x, msg.y, msg.z])

    def rotation_cb(self, msg):
        self.current_rotation = Quat(msg.w, msg.x, msg.y, msg.z)

    # --------- Utility ----------
    def log_message(self, text):
        req = String.Request()
        req.data = text
        self.logger_client.call_async(req)

    def distance(self, a, b):
        return float(np.linalg.norm(a - b))

    def pointing_error(self, pos, target, quat):
        m = quat.to_matrix()
        forward = m[:, 0]
        forward = np.array([forward[0], forward[1], 0])

        desired = target - pos
        desired = np.array([desired[0], desired[1], 0])

        forward = forward / np.linalg.norm(forward)
        desired = desired / np.linalg.norm(desired)

        dot = np.dot(forward, desired)
        dot = np.clip(dot, -1.0, 1.0)

        angle = np.arccos(dot)
        cross_z = forward[0]*desired[1] - forward[1]*desired[0]
        if cross_z < 0:
            angle = -angle

        return angle

    def send_accel(self, value):
        req = Float.Request()
        req.data = float(value)
        self.accel_client.call_async(req)

    def send_brake(self, value):
        req = Float.Request()
        req.data = float(value)
        self.brake_client.call_async(req)

    def send_steer(self, value):
        req = Float.Request()
        req.data = float(value)
        self.steer_client.call_async(req)

    def send_core_sample(self):
        req = Float.Request()
        req.data = 0.0
        self.core_sampling_client.call_async(req)

    # --------- MAIN LOOP ----------
    def timer_callback(self):
        if (self.current_location is None or
            self.current_rotation is None or
            self.current_velocity is None):
            return

        pos = self.current_location
        target_index = self.waypoint_order[self.current_wp_idx]
        target = self.waypoints[target_index]


        dist = self.distance(pos, target)
        heading_err = self.pointing_error(pos, target, self.current_rotation)

        # ---- Compute steering ----
        steer_cmd = np.clip(heading_err / (0.25 * math.pi), -1.0, 1.0)

        # ---- Simple acceleration ----
        accel_cmd = 1.0 if dist > 40 else 0.3

        # ---- Log ONLY what you asked ----
        self.log_message(
            f"Position: ({pos[0]:.2f}, {pos[1]:.2f}), "
            f"Distance: {dist:.2f}, "
            f"Heading error: {math.degrees(heading_err):.1f} deg, "
            f"Steer: {steer_cmd:.2f}, "
            f"Accel: {accel_cmd:.2f}, "
            f"Speed: {np.linalg.norm(self.current_velocity):.2f} m/s, "
            f"Waypoint: {self.current_wp_idx}"
        )

        # ---- Reached waypoint → do core sampling ----
        if dist < self.tolerance:

            self.log_message(
                f"Target reached! Beginning core sampling at ({pos[0]:.2f}, {pos[1]:.2f})"
            )
            self.send_core_sample()

            # Move to next waypoint
            if self.current_wp_idx == len(self.waypoints) - 1:
                self.log_message("All waypoints reached. Mission complete.")
                return

            self.current_wp_idx += 1
            next_index = self.waypoint_order[self.current_wp_idx]
            nxt = self.waypoints[next_index]

            self.log_message(
                f"After sampling, moving to next waypoint: ({nxt[0]:.2f}, {nxt[1]:.2f})"
            )
            return

        


def main(args=None):
    rclpy.init(args=args)
    node = LoggingRover()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
