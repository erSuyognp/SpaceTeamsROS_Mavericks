#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from space_teams_definitions.srv import String, Float
from geometry_msgs.msg import Point, Quaternion
import math
import time
from space_teams_python.transformations import *


class RoverController(Node):
    def __init__(self):
        super().__init__('RoverController')

        # Create clients for only the needed services
        self.logger_client = self.create_client(String, 'log_message')
        self.accelerator_client = self.create_client(Float, 'Accelerator')
        self.reverse_client = self.create_client(Float, 'Reverse')

        # Print "Hello" once when starting
        self.log_message("Hello")

        # Set up a timer to alternate between forward and reverse
        self.forward = True
        self.timer = self.create_timer(10.0, self.timer_callback)  # every 3 seconds

        self.get_logger().info("Minimal Rover Controller started.")

    def log_message(self, message: str):
        """Send a log message to /log_message."""
        request = String.Request()
        request.data = message
        self.logger_client.call_async(request)

    def send_accelerator_command(self, value: float):
        """Send accelerator command (0–1)."""
        request = Float.Request()
        request.data = max(0.0, min(1.0, value))
        self.accelerator_client.call_async(request)

    def send_reverse_command(self, value: float):
        """Send reverse command (0–1)."""
        request = Float.Request()
        request.data = max(0.0, min(1.0, value))
        self.reverse_client.call_async(request)

    def timer_callback(self):
        """Alternate between forward and reverse motion."""
        self.log_message("Hello")
        if self.forward:
            self.log_message("Moving forward")
            self.send_accelerator_command(0.5)
            self.send_reverse_command(0.0)
        else:
            self.log_message("Reversing")
            self.send_accelerator_command(0.0)
            self.send_reverse_command(0.5)

        self.forward = not self.forward


def main(args=None):
    rclpy.init(args=args)
    node = RoverController()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
