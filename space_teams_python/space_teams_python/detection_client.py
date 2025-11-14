#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import os

from std_msgs.msg import Float32MultiArray

from dataclasses import dataclass
import rclpy.qos
import cv2
from space_teams_python.utils import direction_manuver, detect_rock

@dataclass
class TopicConfig:
    name: str
    qos_history_depth: int = 5  # Keep more messages in history
    qos_reliability: int = rclpy.qos.ReliabilityPolicy.BEST_EFFORT  # Best effort for real-time
    qos_durability: int = rclpy.qos.DurabilityPolicy.VOLATILE  # No need to store

class detection(Node):
    def __init__(self):
        super().__init__('image_saver')

        # Declare parameters
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('save_directory', './saved_images')
        self.declare_parameter('image_format', 'jpg')
        self.declare_parameter('depth_format', 'png')

        # Get parameters
        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.save_directory = self.get_parameter('save_directory').get_parameter_value().string_value
        self.image_format = self.get_parameter('image_format').get_parameter_value().string_value
        self.depth_format = self.get_parameter('depth_format').get_parameter_value().string_value

        # Create save directory if it doesn't exist
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Subscribe to the image topics
        qos_rgb = self._create_qos_profile(TopicConfig(name='camera/image_raw'))
        qos_depth = self._create_qos_profile(TopicConfig(name='camera/depth/image_raw'))
        self.image_subscription = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            qos_profile=qos_rgb
        )
        # self.depth_subscription = self.create_subscription(
        #     Image,
        #     self.depth_topic,
        #     self.depth_callback,
        #     qos_profile=qos_depth
        # )
        self.obstacle_detection_publisher = self.create_publisher(Float32MultiArray, 'obstacle_detections', 10)

        self.get_logger().info(f"Subscribed to image topic: {self.image_topic}")
        self.get_logger().info(f"Subscribed to depth topic: {self.depth_topic}")
        self.get_logger().info(f"Images will be saved to: {self.save_directory}")

        self.image_count = 0
        self.depth_count = 0

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            self.get_logger().info("Received image for processing.")    
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Generate file name
            file_name = f"image_{self.image_count:04d}.{self.image_format}"
            file_path = os.path.join(self.save_directory, file_name)

            # Save the image
            valid_rocks = detect_rock(cv_image)

            if valid_rocks:
                for valid_rock in valid_rocks:
                    x, y, w, h = valid_rock['bbox']
                    cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            cv2.imshow("Detected Rocks", cv_image)
            obstacle_array = direction_manuver(detect_rock(cv_image))
            obstacle_msg = Float32MultiArray()
            obstacle_msg.data = obstacle_array
            self.obstacle_detection_publisher.publish(obstacle_msg)
            self.get_logger().info(f"Published obstacle info: {obstacle_array}")
            
        except Exception as e:
            self.get_logger().error(f"Failed to publish obstacle info: {e}")

    def depth_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            # Generate file name
            file_name = f"depth_{self.depth_count:04d}.{self.depth_format}"
            file_path = os.path.join(self.save_directory, file_name)

            # Save the depth image
            # cv2.imwrite(file_path, depth_image)
            # self.get_logger().info(f"Saved depth image: {file_path}")

            self.depth_count += 1
        except Exception as e:
            self.get_logger().error(f"Failed to save depth image: {e}")
    
    def _create_qos_profile(self, topic_config: TopicConfig) -> rclpy.qos.QoSProfile:
        """Create QoS profile for image publishers."""
        return rclpy.qos.QoSProfile(
            reliability=topic_config.qos_reliability,
            durability=topic_config.qos_durability,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth=topic_config.qos_history_depth,
            # Tighter deadlines for real-time performance
            deadline=rclpy.duration.Duration(seconds=1/60.0),
            lifespan=rclpy.duration.Duration(seconds=1/30.0),
            liveliness=rclpy.qos.LivelinessPolicy.AUTOMATIC,
            liveliness_lease_duration=rclpy.duration.Duration(seconds=0.1)  # Faster liveliness checks
        )


def main(args=None):
    rclpy.init(args=args)
    detection_client = detection()
    try:
        rclpy.spin(detection_client)
    except KeyboardInterrupt:
        pass
    finally:
        detection_client.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()