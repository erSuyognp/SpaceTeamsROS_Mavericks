#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from space_teams_definitions.srv import String, Float
from geometry_msgs.msg import Point, Quaternion
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import math
import cv2
import time
import numpy as np
from space_teams_python.transformations import *
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data

# Try to import YOLO - graceful degradation if not available
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not available. YOLO rock detection will be disabled.")
    print("Install with: pip install ultralytics")


class RoverController(Node):
    def __init__(self):
        super().__init__('RoverController')
        self.bridge = CvBridge()

        # Define QoS profiles
        # Use sensor_data QoS for camera topics (more permissive)
        camera_qos_profile = qos_profile_sensor_data
        # Standard QoS for other topics
        standard_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Service clients
        self.logger_client = self.create_client(String, 'log_message')
        self.steer_client = self.create_client(Float, 'Steer')
        self.accelerator_client = self.create_client(Float, 'Accelerator')
        self.reverse_client = self.create_client(Float, 'Reverse')
        self.brake_client = self.create_client(Float, 'Brake')
        self.core_sampling_client = self.create_client(Float, 'CoreSample')
        self.change_exposure_client = self.create_client(Float, 'ChangeExposure')

        # Topic subscriptions
        self.current_location_marsFrame = None
        self.current_velocity_marsFrame = None
        self.current_rotation_marsFrame = None
        self.current_location_localFrame = None
        self.current_velocity_localFrame = None
        self.current_rotation_localFrame = None
        self.state = "Driving"

        self.create_subscription(Point, 'LocationMarsFrame', self.location_marsFrame_callback, 10)
        self.create_subscription(Point, 'VelocityMarsFrame', self.velocity_marsFrame_callback, 10)
        self.create_subscription(Quaternion, 'RotationMarsFrame', self.rotation_marsFrame_callback, 10)
        self.create_subscription(Point, 'LocationLocalFrame', self.location_localFrame_callback, 10)
        self.create_subscription(Point, 'VelocityLocalFrame', self.velocity_localFrame_callback, 10)
        self.create_subscription(Quaternion, 'RotationLocalFrame', self.rotation_localFrame_callback, 10)

        self.create_subscription(Point, 'CoreSamplingComplete', self.core_sampling_complete_callback, 1)

        # Camera subscriptions with improved QoS and encoding handling
        self.create_subscription(Image, '/camera/image_raw', self.image_callback, camera_qos_profile)   
        self.create_subscription(Image, '/camera/depth/image_raw', self.depth_callback, camera_qos_profile)

        # Control state
        self.target_loc_localFrame = None
        self.tolerance = 5.0  # meters
        self.max_speed = 0.5
        self.navigation_active = False
        self.navigation_iterations = 0
        self.initial_move_end_time = None
        self.initial_move_done = False

        # Waypoints
        self.waypoints = None
        self.current_waypoint_idx = None

        # Camera state tracking
        self.latest_rgb = None
        self.latest_depth = None
        self.rgb_frame_count = 0
        self.depth_frame_count = 0
        self.last_rgb_time = None
        self.last_depth_time = None

        # Obstacle detection parameters
        self.obstacle_detection_enabled = True
        self.obstacle_threshold_distance = 200.0  # meters - obstacle if closer than this
        self.obstacle_detection_region_height = 0.4  # Use bottom 40% of image for obstacle detection
        self.obstacle_avoidance_active = False
        self.obstacle_clearance_time = None
        self.obstacle_clearance_duration = 2.0  # seconds to wait after obstacle cleared
        self.avoidance_steer_direction = 0.0  # -1.0 (left) to 1.0 (right)

        # YOLO rock detection parameters
        self.yolo_enabled = YOLO_AVAILABLE
        self.yolo_model = None
        self.rock_detections = []  # List of detected rocks: [(x_center, y_center, width, height, confidence), ...]
        self.rock_detection_confidence_threshold = 0.5  # Minimum confidence for rock detection
        self.rock_danger_zone_threshold = 0.4  # Rocks in bottom 40% of image are considered dangerous
        self.yolo_model_path = 'yolov8n.pt'  # Default YOLOv8 nano model (smallest, fastest)
        
        # Initialize YOLO model if available
        if self.yolo_enabled:
            try:
                self.yolo_model = YOLO(self.yolo_model_path)
                self.get_logger().info(f"YOLO model loaded: {self.yolo_model_path}")
                self.get_logger().info("Rock detection enabled. Looking for: person, bicycle, car, motorcycle, bus, truck (as rock proxies)")
            except Exception as e:
                self.get_logger().error(f"Failed to load YOLO model: {e}")
                self.yolo_enabled = False
        else:
            self.get_logger().warn("YOLO not available - rock detection disabled. Install with: pip install ultralytics")

        # Stuck detection parameters
        self.stuck_detection_enabled = True
        self.stuck_threshold_distance = 1.0  # meters - considered stuck if moved less than this
        self.stuck_time_threshold = 25.0  # seconds - stuck if not moved for this long
        self.stuck_check_position = None
        self.stuck_check_time = None
        self.is_stuck = False
        self.getting_unstuck = False
        self.unstuck_start_time = None
        self.unstuck_duration = 5.0  # seconds to reverse and turn left
        self.unstuck_reverse_duration = 5.0  # seconds to reverse
        self.unstuck_turn_duration = 1.0  # seconds to turn left after reversing

        # Timer for control loop
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.get_logger().info('Rover controller is ready.')

    def location_marsFrame_callback(self, msg):
        self.current_location_marsFrame = msg
    
    def velocity_marsFrame_callback(self, msg):
        self.current_velocity_marsFrame = msg

    def rotation_marsFrame_callback(self, msg):
        self.current_rotation_marsFrame = msg

    def location_localFrame_callback(self, msg):
        self.current_location_localFrame = msg
        
        # Update stuck detection check position
        if self.stuck_detection_enabled and self.navigation_active:
            current_pos = np.array([float(msg.x), float(msg.y), float(msg.z)])
            
            # Initialize or check if moved significantly
            if self.stuck_check_position is None:
                self.stuck_check_position = current_pos
                self.stuck_check_time = time.time()
                self.is_stuck = False
            else:
                distance_moved = np.linalg.norm(current_pos[:2] - self.stuck_check_position[:2])
                
                if distance_moved > self.stuck_threshold_distance:
                    # Moved significantly, reset stuck detection
                    self.stuck_check_position = current_pos
                    self.stuck_check_time = time.time()
                    if self.is_stuck:
                        self.is_stuck = False
                        self.get_logger().info("Rover is no longer stuck - movement detected")
                else:
                    # Check if stuck for too long
                    time_stuck = time.time() - self.stuck_check_time
                    if time_stuck > self.stuck_time_threshold and not self.is_stuck:
                        self.is_stuck = True
                        self.get_logger().warn(f"Rover is STUCK! Hasn't moved {distance_moved:.2f}m in {time_stuck:.1f} seconds")
                        self.log_message(f"STUCK DETECTED: Rover hasn't moved in {time_stuck:.1f} seconds. Initiating recovery maneuver.")

    def velocity_localFrame_callback(self, msg):
        self.current_velocity_localFrame = msg

    def rotation_localFrame_callback(self, msg):
        self.current_rotation_localFrame = msg
    
    def core_sampling_complete_callback(self, msg):
        self.state = "Driving"

    def image_callback(self, msg):
        try:
            self.rgb_frame_count += 1
            self.last_rgb_time = time.time()
            
            # Log first frame
            if self.rgb_frame_count == 1:
                self.get_logger().info(f"Received first RGB frame: {msg.width}x{msg.height}, encoding: {msg.encoding}")
            
            # Convert ROS Image message to OpenCV image
            # Note: image_client publishes as 'rgb8', but OpenCV uses 'bgr8' for display
            if msg.encoding == 'rgb8':
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
                # Convert RGB to BGR for OpenCV display
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            else:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            self.latest_rgb = cv_image

            # Process the image (YOLO rock detection)
            self.process_image(cv_image)

            # Draw YOLO detections on image
            if self.yolo_enabled and self.rock_detections:
                self.draw_rock_detections(cv_image)

            # Display the camera feed
            cv2.imshow('Camera Feed', cv_image)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Error processing RGB image: {str(e)}')
            import traceback
            self.get_logger().error(traceback.format_exc())

    def process_image(self, cv_image):
        """
        Process incoming RGB images with YOLO for rock detection.
        """
        if not self.yolo_enabled or self.yolo_model is None:
            return
        
        try:
            # Run YOLO inference
            # Note: YOLO expects BGR format (which cv_image already is)
            results = self.yolo_model(cv_image, verbose=False, conf=self.rock_detection_confidence_threshold)
            
            # Clear previous detections
            self.rock_detections = []
            
            # Process detections
            height, width = cv_image.shape[:2]
            danger_zone_y = int(height * (1.0 - self.rock_danger_zone_threshold))
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get class name and confidence
                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = result.names[cls_id]
                    
                    # Filter for objects that could be rocks/obstacles
                    # Using common COCO classes that might represent rocks/obstacles
                    # You can customize this list based on your specific needs
                    rock_proxy_classes = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 
                                         'backpack', 'umbrella', 'handbag', 'suitcase', 'sports ball']
                    
                    if class_name.lower() in rock_proxy_classes or confidence > 0.7:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x_center = (x1 + x2) / 2
                        y_center = (y1 + y2) / 2
                        box_width = x2 - x1
                        box_height = y2 - y1
                        
                        # Store detection
                        self.rock_detections.append({
                            'x_center': x_center,
                            'y_center': y_center,
                            'width': box_width,
                            'height': box_height,
                            'confidence': confidence,
                            'class': class_name,
                            'in_danger_zone': y_center > danger_zone_y
                        })
            
            # Check if rocks detected in danger zone
            dangerous_rocks = [r for r in self.rock_detections if r['in_danger_zone']]
            if dangerous_rocks:
                self.process_rock_obstacles(dangerous_rocks, width)
            
        except Exception as e:
            self.get_logger().error(f'Error in YOLO rock detection: {str(e)}')
            import traceback
            self.get_logger().error(traceback.format_exc())

    def process_rock_obstacles(self, dangerous_rocks, image_width):
        """
        Process detected rocks in the danger zone and trigger obstacle avoidance.
        """
        if not dangerous_rocks:
            return
        
        # Divide image into left, center, right regions
        region_width = image_width // 3
        left_rocks = []
        center_rocks = []
        right_rocks = []
        
        for rock in dangerous_rocks:
            x_center = rock['x_center']
            if x_center < region_width:
                left_rocks.append(rock)
            elif x_center < 2 * region_width:
                center_rocks.append(rock)
            else:
                right_rocks.append(rock)
        
        # Determine avoidance direction based on rock locations
        if center_rocks or (left_rocks and right_rocks):
            # Rocks in center or both sides - choose direction with fewer rocks
            if len(left_rocks) < len(right_rocks):
                self.avoidance_steer_direction = -1.0  # Steer left
            else:
                self.avoidance_steer_direction = 1.0  # Steer right
            self.obstacle_avoidance_active = True
            self.obstacle_clearance_time = None
            if self.depth_frame_count % 10 == 0:  # Log periodically
                self.get_logger().warn(f"YOLO: Rocks detected in danger zone! L:{len(left_rocks)} C:{len(center_rocks)} R:{len(right_rocks)}")
        elif left_rocks:
            # Rocks on left - steer right
            self.avoidance_steer_direction = 1.0
            self.obstacle_avoidance_active = True
            self.obstacle_clearance_time = None
            if self.depth_frame_count % 10 == 0:
                self.get_logger().warn(f"YOLO: {len(left_rocks)} rock(s) detected on LEFT - steering RIGHT")
        elif right_rocks:
            # Rocks on right - steer left
            self.avoidance_steer_direction = -1.0
            self.obstacle_avoidance_active = True
            self.obstacle_clearance_time = None
            if self.depth_frame_count % 10 == 0:
                self.get_logger().warn(f"YOLO: {len(right_rocks)} rock(s) detected on RIGHT - steering LEFT")

    def depth_callback(self, msg):
        try:
            self.depth_frame_count += 1
            self.last_depth_time = time.time()
            
            # Log first frame
            if self.depth_frame_count == 1:
                self.get_logger().info(f"Received first Depth frame: {msg.width}x{msg.height}, encoding: {msg.encoding}")
            
            # Convert to numpy array (depth map)
            # image_client publishes depth as '32FC1' (32-bit float, single channel)
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.latest_depth = depth_image
            
            # Perform obstacle detection
            if self.obstacle_detection_enabled:
                self.detect_obstacles(depth_image)
            
            # You can access distance values directly from the image
            # For example, to get the distance at the center:
            height, width = depth_image.shape
            center_distance = float(depth_image[height//2, width//2])
            
            # Log center distance periodically (every 30 frames to reduce spam)
            if self.depth_frame_count % 30 == 0:
                self.get_logger().info(f'Center distance: {center_distance:.2f} meters')
            
            # Visualize the depth map with obstacle regions
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), 
                cv2.COLORMAP_JET
            )
            
            # Draw obstacle detection regions on visualization
            if self.obstacle_detection_enabled and depth_image is not None:
                self.visualize_obstacle_regions(depth_colormap, depth_image)
            
            cv2.imshow('Depth Camera', depth_colormap)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {str(e)}')
            import traceback
            self.get_logger().error(traceback.format_exc())

    def process_combined(self):
        """
        Example of combining RGB and Depth frames.
        You can correlate pixels between RGB and depth images to get rich information about the environment.
        This enables applications like semantic segmentation with distance awareness or object detection with distance estimation.
        """
        if self.latest_rgb is not None and self.latest_depth is not None:
            # You could align them or extract pixel distance, etc.
            h, w, _ = self.latest_rgb.shape
            center_dist = self.latest_depth[h // 2, w // 2]
            # Example: Log combined information (can be removed or made less frequent)
            # self.get_logger().info(f'RGB+Depth center distance: {center_dist:.2f} m')

    def detect_obstacles(self, depth_image):
        """
        Detect obstacles in the depth image by analyzing regions.
        Divides the image into left, center, and right regions and checks for obstacles.
        """
        if depth_image is None:
            return
        
        height, width = depth_image.shape
        
        # Use bottom portion of image for obstacle detection (closer to ground)
        detection_start_row = int(height * (1.0 - self.obstacle_detection_region_height))
        detection_region = depth_image[detection_start_row:, :]
        
        # Divide into three regions: left, center, right
        region_width = width // 3
        left_region = detection_region[:, :region_width]
        center_region = detection_region[:, region_width:2*region_width]
        right_region = detection_region[:, 2*region_width:]
        
        # Calculate median distance for each region (more robust than mean)
        def safe_median(region):
            # Filter out invalid values (NaN, Inf, zero)
            valid_values = region[(region > 0.1) & (region < 100.0) & np.isfinite(region)]
            if len(valid_values) > 0:
                return np.median(valid_values)
            return float('inf')
        
        left_distance = safe_median(left_region)
        center_distance = safe_median(center_region)
        right_distance = safe_median(right_region)
        
        # Check for obstacles (distance < threshold)
        left_obstacle = left_distance < self.obstacle_threshold_distance
        center_obstacle = center_distance < self.obstacle_threshold_distance
        right_obstacle = right_distance < self.obstacle_threshold_distance
        
        # Determine avoidance direction
        if center_obstacle or (left_obstacle and right_obstacle):
            # Obstacle in center or both sides - choose direction with more clearance
            if left_distance > right_distance:
                self.avoidance_steer_direction = -1.0  # Steer left
            else:
                self.avoidance_steer_direction = 1.0  # Steer right
            self.obstacle_avoidance_active = True
            self.obstacle_clearance_time = None
        elif left_obstacle:
            # Obstacle on left - steer right
            self.avoidance_steer_direction = 1.0
            self.obstacle_avoidance_active = True
            self.obstacle_clearance_time = None
        elif right_obstacle:
            # Obstacle on right - steer left
            self.avoidance_steer_direction = -1.0
            self.obstacle_avoidance_active = True
            self.obstacle_clearance_time = None
        else:
            # No obstacles detected
            if self.obstacle_avoidance_active:
                # Start clearance timer
                if self.obstacle_clearance_time is None:
                    self.obstacle_clearance_time = time.time()
                elif time.time() - self.obstacle_clearance_time > self.obstacle_clearance_duration:
                    # Obstacle cleared, resume normal navigation
                    self.obstacle_avoidance_active = False
                    self.obstacle_clearance_time = None
                    self.avoidance_steer_direction = 0.0
                    self.get_logger().info("Obstacle cleared, resuming normal navigation")
        
        # Log obstacle status periodically
        if self.depth_frame_count % 30 == 0 and self.obstacle_avoidance_active:
            self.get_logger().warn(
                f"Obstacle detected! L:{left_distance:.1f}m C:{center_distance:.1f}m R:{right_distance:.1f}m "
                f"Steering: {'LEFT' if self.avoidance_steer_direction < 0 else 'RIGHT'}"
            )

    def draw_rock_detections(self, cv_image):
        """
        Draw YOLO rock detections on the camera feed.
        """
        if not self.rock_detections:
            return
        
        height, width = cv_image.shape[:2]
        danger_zone_y = int(height * (1.0 - self.rock_danger_zone_threshold))
        
        # Draw danger zone line
        cv2.line(cv_image, (0, danger_zone_y), (width, danger_zone_y), (0, 255, 255), 2)
        cv2.putText(cv_image, 'DANGER ZONE', (10, danger_zone_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw each detection
        for rock in self.rock_detections:
            x_center = int(rock['x_center'])
            y_center = int(rock['y_center'])
            w = int(rock['width'])
            h = int(rock['height'])
            conf = rock['confidence']
            class_name = rock['class']
            in_danger = rock['in_danger_zone']
            
            # Color: red for danger zone, yellow for safe zone
            color = (0, 0, 255) if in_danger else (0, 255, 255)
            thickness = 3 if in_danger else 2
            
            # Draw bounding box
            x1 = int(x_center - w/2)
            y1 = int(y_center - h/2)
            x2 = int(x_center + w/2)
            y2 = int(y_center + h/2)
            cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            label = f"{class_name}: {conf:.2f}"
            if in_danger:
                label += " [DANGER]"
            cv2.putText(cv_image, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw center point
            cv2.circle(cv_image, (x_center, y_center), 5, color, -1)

    def visualize_obstacle_regions(self, depth_colormap, depth_image):
        """
        Draw obstacle detection regions on the depth visualization.
        """
        if depth_image is None:
            return
        
        height, width = depth_image.shape
        detection_start_row = int(height * (1.0 - self.obstacle_detection_region_height))
        region_width = width // 3
        
        # Draw region boundaries
        cv2.line(depth_colormap, (region_width, detection_start_row), (region_width, height), (255, 255, 255), 2)
        cv2.line(depth_colormap, (2*region_width, detection_start_row), (2*region_width, height), (255, 255, 255), 2)
        cv2.line(depth_colormap, (0, detection_start_row), (width, detection_start_row), (255, 255, 255), 2)
        
        # Draw labels
        cv2.putText(depth_colormap, 'L', (region_width//2, height-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(depth_colormap, 'C', (region_width + region_width//2, height-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(depth_colormap, 'R', (2*region_width + region_width//2, height-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Highlight if obstacle avoidance is active
        if self.obstacle_avoidance_active:
            overlay = depth_colormap.copy()
            cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 255), 10)  # Red border
            cv2.addWeighted(overlay, 0.3, depth_colormap, 0.7, 0, depth_colormap)
            cv2.putText(depth_colormap, 'OBSTACLE AVOIDANCE ACTIVE', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def log_message(self, message):
        request = String.Request()
        request.data = message
        future = self.logger_client.call_async(request)
        return future

    def send_steer_command(self, steer_value):
        request = Float.Request()
        request.data = max(-1.0, min(1.0, steer_value))
        return self.steer_client.call_async(request)

    def send_accelerator_command(self, accel_value):
        request = Float.Request()
        request.data = max(0.0, min(1.0, accel_value))
        return self.accelerator_client.call_async(request)

    def send_reverse_command(self, reverse_value):
        request = Float.Request()
        request.data = max(0.0, min(1.0, reverse_value))
        return self.reverse_client.call_async(request)

    def send_brake_command(self, brake_value):
        request = Float.Request()
        request.data = max(0.0, min(1.0, brake_value))
        return self.brake_client.call_async(request)
    
    def send_core_sampling_command(self):
        self.state = "Sampling"
        request = Float.Request()
        request.data = 0.0
        return self.core_sampling_client.call_async(request)
    
    def calculate_direction_to_target(self, current_loc_localFrame: npt.NDArray, 
                                      target_loc_localFrame: npt.NDArray) -> npt.NDArray:
        return normalize(target_loc_localFrame - current_loc_localFrame)
    
    def calculate_error_angle_sign(self, vec1: npt.NDArray, vec2: npt.NDArray) -> float:
        return 1.0 if np.dot(np.cross(vec1, vec2), np.array([0.0, 0.0, 1.0])) > 0.0 else -1.0
    
    def error_angle_arctan(self, vec1, vec2):
        up = normalize(np.cross(np.cross(vec2, vec1), vec2))
        x = np.dot(vec1, vec2)
        y = np.dot(vec1, up)
        return np.arctan2(y, x)
    
    def calculate_pointing_error_angle(self, current_loc_localFrame: npt.NDArray, 
                                       target_loc_localFrame: npt.NDArray, current_rot_localFrame: Quat) -> float:
        m = current_rot_localFrame.to_matrix()
        forward = m[:, 0]
        forward = normalize(np.array([forward[0], forward[1], 0.0]))

        target_direction = self.calculate_direction_to_target(current_loc_localFrame, target_loc_localFrame)
        target_direction = normalize(np.array([target_direction[0], target_direction[1], 0.0]))

        # error_angle = self.error_angle_arctan(forward, target_direction)

        error_angle = np.arccos(np.dot(forward, target_direction))
        error_angle_dir = self.calculate_error_angle_sign(target_direction, forward)
        return error_angle_dir * error_angle

    def calculate_distance_to_target(self, current_loc_localFrame: npt.NDArray, target_loc_localFrame: npt.NDArray):
        return np.linalg.norm(target_loc_localFrame - current_loc_localFrame)

    def calculate_speed_difference(self, current_vel_localFrame: npt.NDArray, target_speed_kph: float) -> float:
        return mps_to_kph(kph_to_mps(target_speed_kph) - np.linalg.norm(current_vel_localFrame))

    def start_navigation(self, target_loc_localFrame):
        self.target_loc_localFrame = target_loc_localFrame
        self.navigation_active = True
        self.navigation_iterations = 0
        self.initial_move_done = False
        self.initial_move_end_time = time.time() + 10.0
        
        # Reset stuck detection when starting navigation
        if self.stuck_detection_enabled:
            self.stuck_check_position = None
            self.stuck_check_time = None
            self.is_stuck = False
            self.getting_unstuck = False
        
        self.log_message(f"Starting navigation to target: ({target_loc_localFrame[0]:.2f}, {target_loc_localFrame[1]:.2f})")
        self.send_accelerator_command(0.2)

    def change_exposure(self, exposure_level: float):
        request = Float.Request()
        request.data = exposure_level
        return self.change_exposure_client.call_async(request)

    def check_stuck(self, current_loc_localFrame):
        """
        Check if rover is stuck and initiate recovery if needed.
        """
        if not self.stuck_detection_enabled or not self.navigation_active:
            return False
        
        if self.is_stuck and not self.getting_unstuck:
            # Start getting unstuck maneuver
            self.getting_unstuck = True
            self.unstuck_start_time = time.time()
            self.get_logger().warn("Starting recovery maneuver: reverse and turn left")
            self.log_message("Executing recovery maneuver: reversing and turning left")
            return True
        
        return self.getting_unstuck

    def execute_unstuck_maneuver(self):
        """
        Execute the recovery maneuver: reverse for a few seconds, then turn left.
        """
        if not self.getting_unstuck:
            return False
        
        elapsed = time.time() - self.unstuck_start_time
        
        if elapsed < self.unstuck_reverse_duration:
            # Phase 1: Reverse
            self.send_reverse_command(0.5)  # Moderate reverse speed
            self.send_steer_command(0.0)  # No steering while reversing
            self.send_accelerator_command(0.0)
            self.send_brake_command(0.0)
            if int(elapsed * 2) % 2 == 0:  # Log every 0.5 seconds
                self.get_logger().info(f"Recovery: Reversing... ({elapsed:.1f}s)")
        elif elapsed < self.unstuck_reverse_duration + self.unstuck_turn_duration:
            # Phase 2: Turn left while stopped/reversing slightly
            turn_elapsed = elapsed - self.unstuck_reverse_duration
            self.send_reverse_command(0.0)  # Slight reverse while turning
            self.send_steer_command(-0.8)  # Turn left (negative = left)
            self.send_accelerator_command(0.2)
            self.send_brake_command(0.0)
            if int(turn_elapsed * 2) % 2 == 0:  # Log every 0.5 seconds
                self.get_logger().info(f"Recovery: Turning left... ({turn_elapsed:.1f}s)")
        else:
            # Maneuver complete, reset stuck state
            self.send_reverse_command(0.0)
            self.send_steer_command(0.0)
            self.send_accelerator_command(0.0)
            self.send_brake_command(0.0)
            
            self.getting_unstuck = False
            self.is_stuck = False
            self.stuck_check_position = None  # Reset stuck detection
            self.stuck_check_time = None
            self.unstuck_start_time = None
            
            self.get_logger().info("Recovery maneuver complete. Resuming normal navigation.")
            self.log_message("Recovery maneuver complete. Resuming navigation.")
            return False  # Maneuver complete
        
        return True  # Still executing maneuver

    def timer_callback(self):
        if not self.navigation_active:
            return

        # Initial move forward for 10 seconds
        if not self.initial_move_done and self.initial_move_end_time is not None:
            if time.time() < self.initial_move_end_time:
                return
            self.send_accelerator_command(0.0)
            self.initial_move_done = True
        
        # Navigation logic
        if self.current_location_localFrame is None or self.current_rotation_localFrame is None:
            self.get_logger().info("Waiting for location/rotation update...")
            return

        # Get location
        current_x = float(self.current_location_localFrame.x)
        current_y = float(self.current_location_localFrame.y)
        current_z = float(self.current_location_localFrame.z)
        current_loc_localFrame = np.array([current_x, current_y, current_z])

        # Check for stuck condition and execute recovery if needed
        # This has highest priority - if stuck, execute recovery maneuver
        if self.check_stuck(current_loc_localFrame):
            if self.execute_unstuck_maneuver():
                # Still executing recovery maneuver
                self.navigation_iterations += 1
                return
            # Recovery complete, continue with normal navigation

        # Get velocity
        current_vx = float(self.current_velocity_localFrame.x)
        current_vy = float(self.current_velocity_localFrame.y)
        current_vz = float(self.current_velocity_localFrame.z)
        current_vel_localFrame = np.array([current_vx, current_vy, current_vz])

        # Get rotation
        qx = float(self.current_rotation_localFrame.x)
        qy = float(self.current_rotation_localFrame.y)
        qz = float(self.current_rotation_localFrame.z)
        qw = float(self.current_rotation_localFrame.w)
        current_rot_localFrame = Quat(qw, qx, qy, qz)

        # Distance to target
        distance = self.calculate_distance_to_target(current_loc_localFrame, self.target_loc_localFrame)
        if distance < self.tolerance:
            self.send_brake_command(1.0)
            self.send_steer_command(0.0)
            self.send_accelerator_command(0.0)
            self.log_message(f"Target reached! Beginning core sampling at position: ({current_x:.2f}, {current_y:.2f})")
            self.send_core_sampling_command()

            if self.current_waypoint_idx == len(self.waypoints) - 1:
                self.navigation_active = False
                self.log_message("Navigation complete: all waypoints reached and all core samples collected.")
            else:
                self.current_waypoint_idx += 1
                self.target_loc_localFrame = self.waypoints[self.current_waypoint_idx]
                next_loc = f"({self.target_loc_localFrame[0]:.2f}, {self.target_loc_localFrame[1]:.2f})"
                self.log_message(f"After sampling, moving to next waypoint at: {next_loc}")
            return
        
        # Velocity error
        speed_limit_kph = 15.0
        speed_diff_kph = self.calculate_speed_difference(current_vel_localFrame, speed_limit_kph)  # target - current
        accel_factor = remap_clamp(0.0, speed_limit_kph, 0.0, 1.0, speed_diff_kph)  # 1 if not moving, 0 if too fast
        brake_factor = 1.0 - remap_clamp(-speed_limit_kph, 0.0, 0.0, 1.0, speed_diff_kph)  # 0 if <= speed limit, 1 if 2x over

        # Obstacle avoidance takes priority over normal navigation
        if self.obstacle_avoidance_active:
            # Obstacle avoidance mode - steer away from obstacle
            avoidance_steer_gain = 0.8  # Strong steering to avoid obstacles
            actual_steer_command = avoidance_steer_gain * self.avoidance_steer_direction
            
            # Reduce speed when avoiding obstacles
            accel_command = 0.3  # Slow down during avoidance
            brake_command = 0.0
            
            self.send_steer_command(actual_steer_command)
            self.send_accelerator_command(accel_command)
            self.send_reverse_command(brake_command)
            self.send_brake_command(0.0)
            
            # Log avoidance action periodically
            if self.navigation_iterations % 10 == 0:
                self.log_message(f"Obstacle avoidance: steering {'LEFT' if actual_steer_command < 0 else 'RIGHT'}")
            
            self.navigation_iterations += 1
            return
        
        # Normal navigation - calculate heading to target
        # Heading error
        db_heading = np.deg2rad(3.0)  # deadband for heading alignment
        heading_error = self.calculate_pointing_error_angle(current_loc_localFrame, self.target_loc_localFrame, 
                                                            current_rot_localFrame)
        
        # Steering
        steer_command = remap_clamp(-0.25 * np.pi, 0.25 * np.pi, -1.0, 1.0, heading_error)
        if abs(heading_error) < db_heading:
            steer_command = 0.0
        steer_gain = 1.0
        actual_steer_command = -steer_gain * steer_command
        
        # Acceleration
        accel_gain = 2.0
        accel_command = accel_gain * remap_clamp(0.0, 1.0, accel_factor, accel_factor * 0.5, abs(steer_command))

        # Braking
        # If brake, brake_command > 0.5 results in braking (i.e., boolean behavior)
        # If reverse, float value between 0 and 1 is passed, acts as a gradual deceleration
        brake_gain = 1.0
        brake_command = brake_gain * brake_factor

        self.send_steer_command(actual_steer_command)
        self.send_accelerator_command(accel_command)
        self.send_reverse_command(brake_command)  # Send brake command as a float (reverse)
        self.send_brake_command(0.0)
        # self.send_brake_command(brake_command)  # Send brake command as a bool

        # Print commands for debugging:
        # if self.navigation_iterations % 10 == 0:
        #     self.log_message(
        #         f"Position: ({current_x:.2f}, {current_y:.2f}), "
        #         f"Distance: {distance:.2f}, "
        #         f"Heading error: {math.degrees(heading_error):.1f} deg, "
        #         f"Steer: {steer_command:.2f}, "
        #         f"Accel: {accel_command:.2f}"
        #     )
        self.navigation_iterations += 1


def main(args=None):
    rclpy.init(args=args)
    rover_controller = RoverController()

    # Test waypoint:
    # waypoint_marsframe = np.array([2193073.87847882, 743984.99629174, -2485667.65565136])
    # waypoint_localframe = np.array([22.0285988, 60.41062071, -4.50449595])

    # Test multiple waypoints:
    waypoints_localFrame = [
        np.array([-54.31019727, 191.84449903, -19.54598818]),
        np.array([111.24089259, 427.56166121, -54.81398767]),
        np.array([-349.10709106,  558.01869306, -68.71836618]),
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

    # Wait for initial location and rotation
    while rclpy.ok():
        if rover_controller.current_location_localFrame is not None and rover_controller.current_rotation_localFrame is not None:
            break
        rover_controller.get_logger().info('Waiting for initial location and rotation...')
        rclpy.spin_once(rover_controller, timeout_sec=0.5)

    current_x = rover_controller.current_location_localFrame.x
    current_y = rover_controller.current_location_localFrame.y
    
    waypoint_order = [1, 17, 8, 4, 19, 3, 12, 6, 14, 9, 0, 2, 5, 13, 15, 11, 7, 16]
    waypoints_ordered = [waypoints_localFrame[i] for i in waypoint_order]
    rover_controller.waypoints = waypoints_ordered
    rover_controller.current_waypoint_idx = 0

    rover_controller.log_message(
        f"Starting navigation: moving from ({current_x:.2f}, {current_y:.2f}) to ({waypoints_localFrame[0][0]:.2f}, {waypoints_localFrame[0][1]:.2f})"
    )
    
    rover_controller.start_navigation(waypoints_ordered[0])

    try:
        rclpy.spin(rover_controller)
    except KeyboardInterrupt:
        rover_controller.get_logger().info("Shutting down rover controller...")
    finally:
        rover_controller.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
