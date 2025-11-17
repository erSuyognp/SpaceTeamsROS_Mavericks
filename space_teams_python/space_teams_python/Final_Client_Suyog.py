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
        
        # Frame skipping for performance
        self.yolo_process_every_n_frames = 10  # Process YOLO every 10nth frame
        self.depth_process_every_n_frames = 10  # Process depth obstacle detection every 10nth frame

        # Obstacle detection parameters
        self.obstacle_detection_enabled = True
        self.obstacle_threshold_distance = 1000.0  # meters - obstacle if closer than this (legacy)
        self.obstacle_detection_region_height = 0.3  # Use bottom 30% of image for obstacle detection
        self.obstacle_avoidance_active = False
        self.obstacle_clearance_time = None
        self.obstacle_clearance_duration = 2.0  # seconds to wait after obstacle cleared
        self.avoidance_steer_direction = 0.0  # -1.0 (left) to 1.0 (right) - proportional value
        
        # Enhanced depth processing parameters
        self.stopping_distance = 2.5  # meters - distance threshold for stopping
        self.emergency_distance = 1.0  # meters - emergency stop distance
        self.slow_distance = 1.5  # meters - distance threshold for slowing down
        self.max_range = 100.0  # meters - maximum valid depth range
        self.use_bilateral_filter = True  # Use bilateral filter (preserves edges) vs median filter
        self.bilateral_d = 5  # Bilateral filter diameter
        self.bilateral_sigma_color = 50.0  # Bilateral filter sigma for color space
        self.bilateral_sigma_space = 50.0  # Bilateral filter sigma for coordinate space
        self.median_kernel_size = 5  # Median filter kernel size
        self.morphology_kernel_size = 5  # Morphological operations kernel size
        self.min_blob_area = 50  # Minimum blob area in pixels to consider as obstacle
        self.roi_lower_fraction = 0.4  # Use lower 40% of image (near robot)
        self.roi_center_band_fraction = 0.2  # Central horizontal band (20% of height)
        self.ground_removal_enabled = False  # Enable ground plane removal (requires point cloud)
        # Note: Ground plane removal would require:
        # - Camera intrinsics (fx, fy, cx, cy) to convert depth -> XYZ point cloud
        # - RANSAC for plane fitting to remove ground
        # - DBSCAN for clustering after ground removal
        # See detect_obstacles_with_ground_removal() method (not implemented) for future enhancement
        self.steering_gain = 0.5  # Proportional gain for steering based on lateral offset
        
        # Visualization data
        self.last_obstacle_mask = None
        self.last_obstacle_blobs = []
        self.emergency_stop_active = False

        # YOLO rock detection parameters
        self.yolo_enabled = YOLO_AVAILABLE
        self.yolo_model = None
        self.rock_detections = []  # List of detected rocks: [(x_center, y_center, width, height, confidence), ...]
        self.rock_detection_confidence_threshold = 0.5  # Minimum confidence for rock detection
        self.rock_danger_zone_threshold = 0.5  # Rocks in bottom 50% of image are considered dangerous
        self.yolo_model_path = 'aug_best.pt'  # Default YOLOv8 nano model (smallest, fastest)
        
        # Initialize YOLO model if available
        if self.yolo_enabled:
            try:
                self.yolo_model = YOLO(self.yolo_model_path)
                self.get_logger().info(f"YOLO model loaded: {self.yolo_model_path}")
                self.get_logger().info("Rock detection enabled.")
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

        # Automatic exposure adjustment parameters
        self.TARGET_BRIGHTNESS = 100  # mid-brightness (0-255)
        self.TOLERANCE = 10  # brightness tolerance
        self.current_exposure = 13  # start from some reasonable value

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
            
            # Step 1: Measure the brightness of the image
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            brightness = gray.mean()  # Value between about 0-255
            
            # Step 2 & 3: Adjust exposure slowly and safely based on brightness
            if brightness < self.TARGET_BRIGHTNESS - self.TOLERANCE:
                self.current_exposure -= 0.02  # brighten
            elif brightness > self.TARGET_BRIGHTNESS + self.TOLERANCE:
                self.current_exposure += 0.02  # darken
            
            # Clamp to safe range (your camera's range may differ)
            self.current_exposure = max(5.0, min(20.0, self.current_exposure))
            
            # Send updated exposure to camera
            self.change_exposure(self.current_exposure)
            
            # Print current exposure periodically (every 30 frames to reduce spam)
            if self.rgb_frame_count % 30 == 0:
                self.get_logger().info(f'Current exposure: {self.current_exposure:.3f}, Brightness: {brightness:.1f}')
            
            self.latest_rgb = cv_image

            # Process YOLO only every Nth frame to reduce lag
            # Always display latest frame but use previous detection results
            if self.rgb_frame_count % self.yolo_process_every_n_frames == 0:
                # Process the image (YOLO rock detection)
                self.process_image(cv_image)
            # Always draw latest detections (from last YOLO processing) on current frame
            if self.yolo_enabled and self.rock_detections:
                self.draw_rock_detections(cv_image)

            # Display the camera feed (always show latest frame for real-time feel)
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
                            'in_danger_zone': y_center >= danger_zone_y
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
        Uses proportional steering based on rock position - only steers enough to avoid.
        """
        if not dangerous_rocks:
            return
        
        center_x = image_width // 2
        left_rocks = []
        right_rocks = []
        
        for rock in dangerous_rocks:
            x_center = rock['x_center']
            if x_center < center_x:
                left_rocks.append(rock)
            else:
                right_rocks.append(rock)
        
        # Calculate proportional steering based on rock positions
        # Rocks closer to center need more steering, rocks further away need less
        min_steer = 0.2  # Minimum steering to avoid
        max_steer = 0.6  # Maximum steering (reduced from 0.8)
        
        if left_rocks and right_rocks:
            # Rocks on both sides - calculate average position and steer away from center
            left_avg_x = np.mean([r['x_center'] for r in left_rocks])
            right_avg_x = np.mean([r['x_center'] for r in right_rocks])
            
            # Determine which side has rocks closer to center
            left_dist_from_center = center_x - left_avg_x  # Distance from center to left rocks
            right_dist_from_center = right_avg_x - center_x  # Distance from center to right rocks
            
            # Steer away from the side with rocks closer to center
            if left_dist_from_center < right_dist_from_center:
                # Left rocks closer to center, steer right
                # Closer to center = more steering needed
                steer_intensity = min_steer + (max_steer - min_steer) * (1.0 - left_dist_from_center / center_x)
                self.avoidance_steer_direction = min(steer_intensity, max_steer)
            else:
                # Right rocks closer to center, steer left
                steer_intensity = min_steer + (max_steer - min_steer) * (1.0 - right_dist_from_center / center_x)
                self.avoidance_steer_direction = -min(steer_intensity, max_steer)
            
            self.obstacle_avoidance_active = True
            self.obstacle_clearance_time = None
            if self.depth_frame_count % 10 == 0:
                self.get_logger().warn(f"YOLO: Rocks both sides! L:{len(left_rocks)} R:{len(right_rocks)} Steer:{self.avoidance_steer_direction:.2f}")
        elif left_rocks:
            # Rocks on left - steer right proportionally
            left_avg_x = np.mean([r['x_center'] for r in left_rocks])
            dist_from_center = center_x - left_avg_x  # Distance from center
            
            # Calculate proportional steering: closer to center = more steering
            # Normalize distance (0 at center, center_x at edge)
            steer_intensity = min_steer + (max_steer - min_steer) * (1.0 - dist_from_center / center_x)
            self.avoidance_steer_direction = min(steer_intensity, max_steer)
            
            self.obstacle_avoidance_active = True
            self.obstacle_clearance_time = None
            if self.depth_frame_count % 10 == 0:
                self.get_logger().warn(f"YOLO: {len(left_rocks)} rock(s) on LEFT - steering RIGHT {self.avoidance_steer_direction:.2f}")
        elif right_rocks:
            # Rocks on right - steer left proportionally
            right_avg_x = np.mean([r['x_center'] for r in right_rocks])
            dist_from_center = right_avg_x - center_x  # Distance from center
            
            # Calculate proportional steering: closer to center = more steering
            steer_intensity = min_steer + (max_steer - min_steer) * (1.0 - dist_from_center / center_x)
            self.avoidance_steer_direction = -min(steer_intensity, max_steer)
            
            self.obstacle_avoidance_active = True
            self.obstacle_clearance_time = None
            if self.depth_frame_count % 10 == 0:
                self.get_logger().warn(f"YOLO: {len(right_rocks)} rock(s) on RIGHT - steering LEFT {self.avoidance_steer_direction:.2f}")

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
            
            # Perform obstacle detection only every Nth frame to reduce lag
            if self.obstacle_detection_enabled and self.depth_frame_count % self.depth_process_every_n_frames == 0:
                self.detect_obstacles(depth_image)
            
            # # You can access distance values directly from the image
            # # For example, to get the distance at the center:
            # height, width = depth_image.shape
            # center_distance = float(depth_image[height//2, width//2])
            
            # # Log center distance periodically (every 30 frames to reduce spam)
            # if self.depth_frame_count % 30 == 0:
            #     self.get_logger().info(f'Center distance: {center_distance:.2f} meters')
            
            # # Visualize the depth map with obstacle regions
            # # Always update visualization with latest frame for real-time feel
            # depth_colormap = cv2.applyColorMap(
            #     cv2.convertScaleAbs(depth_image, alpha=0.03), 
            #     cv2.COLORMAP_JET
            # )
            
            # # Draw obstacle detection regions on visualization (use last detection results)
            # if self.obstacle_detection_enabled and depth_image is not None:
            #     self.visualize_obstacle_regions(depth_colormap, depth_image)
            
            # cv2.imshow('Depth Camera', depth_colormap)
            # cv2.waitKey(1)
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

    def preprocess_depth_image(self, depth_image):
        """
        Preprocess depth image: filter noise and clamp invalid values.
        
        Steps:
        1. Apply median or bilateral filter to remove speckle/sensor noise
        2. Clamp invalid values (0, NaN, > max_range) to inf or create ignore mask
        """
        if depth_image is None:
            return None, None
        
        # Convert to float32 if needed
        if depth_image.dtype != np.float32:
            depth_image = depth_image.astype(np.float32)
        
        # Step 1: Apply noise reduction filter
        if self.use_bilateral_filter:
            # Bilateral filter preserves edges better than median
            filtered = cv2.bilateralFilter(
                depth_image, 
                d=self.bilateral_d,
                sigmaColor=self.bilateral_sigma_color,
                sigmaSpace=self.bilateral_sigma_space
            )
        else:
            # Median filter for speckle noise removal
            filtered = cv2.medianBlur(depth_image, self.median_kernel_size)
        
        # Step 2: Clamp invalid values
        # Create mask for valid values: > 0, < max_range, and finite
        valid_mask = (filtered > 0.0) & (filtered < self.max_range) & np.isfinite(filtered)
        
        # Set invalid values to inf (or NaN for ignore)
        processed = filtered.copy()
        processed[~valid_mask] = np.inf
        
        return processed, valid_mask
    
    def create_roi_mask(self, height, width):
        """
        Create Region of Interest mask: lower part + central horizontal band.
        
        Returns:
            roi_mask: Binary mask (True = ROI, False = ignore)
        """
        roi_mask = np.zeros((height, width), dtype=bool)
        
        # Lower part of image (near robot)
        lower_start = int(height * (1.0 - self.roi_lower_fraction))
        roi_mask[lower_start:, :] = True
        
        # Central horizontal band
        center_band_start = int(height * (0.5 - self.roi_center_band_fraction / 2))
        center_band_end = int(height * (0.5 + self.roi_center_band_fraction / 2))
        roi_mask[center_band_start:center_band_end, :] = True
        
        return roi_mask
    
    def detect_obstacles(self, depth_image):
        """
        Enhanced obstacle detection using depth preprocessing, thresholding, 
        morphological operations, and connected components analysis.
        """
        if depth_image is None:
            return
        
        height, width = depth_image.shape
        image_center_x = width // 2
        
        # Step 1: Preprocess depth image
        processed_depth, valid_mask = self.preprocess_depth_image(depth_image)
        if processed_depth is None:
            return
        
        # Step 2: Create ROI mask
        roi_mask = self.create_roi_mask(height, width)
        
        # Step 3: Depth thresholding - create binary mask where depth < stopping_distance
        obstacle_mask = (processed_depth < self.stopping_distance) & valid_mask & roi_mask
        
        # Step 4: Morphological clean-up
        kernel = np.ones((self.morphology_kernel_size, self.morphology_kernel_size), np.uint8)
        # Close holes (dilate then erode)
        obstacle_mask = cv2.morphologyEx(obstacle_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        # Remove small specks (opening: erode then dilate)
        obstacle_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_OPEN, kernel)
        obstacle_mask = obstacle_mask.astype(bool)
        
        # Step 5: Connected components / clustering
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            obstacle_mask.astype(np.uint8), connectivity=8
        )
        
        # Find blobs and compute their properties
        obstacle_blobs = []
        min_depth_overall = float('inf')
        
        for i in range(1, num_labels):  # Skip label 0 (background)
            area = stats[i, cv2.CC_STAT_AREA]
            if area < self.min_blob_area:
                continue  # Ignore small blobs
            
            # Get blob centroid
            centroid_x = int(centroids[i, 0])
            centroid_y = int(centroids[i, 1])
            
            # Compute nearest depth in this blob
            blob_mask = (labels == i)
            blob_depths = processed_depth[blob_mask & valid_mask]
            
            if len(blob_depths) > 0:
                min_depth = np.min(blob_depths)
                min_depth_overall = min(min_depth_overall, min_depth)
                
                obstacle_blobs.append({
                    'centroid_x': centroid_x,
                    'centroid_y': centroid_y,
                    'area': area,
                    'min_depth': min_depth,
                    'label': i
                })
        
        # Step 6: Decide action
        if min_depth_overall < self.emergency_distance:
            # EMERGENCY STOP
            self.obstacle_avoidance_active = True
            self.emergency_stop_active = True
            self.avoidance_steer_direction = 0.0  # No steering, just stop
            self.obstacle_clearance_time = None
            
            if self.depth_frame_count % 10 == 0:
                self.get_logger().error(
                    f"EMERGENCY STOP! Obstacle at {min_depth_overall:.2f}m < {self.emergency_distance}m"
                )
                self.log_message(f"EMERGENCY STOP: Obstacle detected at {min_depth_overall:.2f}m")
        
        elif obstacle_blobs:
            self.emergency_stop_active = False
            # Check if obstacles are in center ROI and closer than slow_distance
            center_roi_start = int(width * 0.3)
            center_roi_end = int(width * 0.7)
            
            center_obstacles = [
                blob for blob in obstacle_blobs
                if (blob['centroid_x'] >= center_roi_start and 
                    blob['centroid_x'] <= center_roi_end and
                    blob['min_depth'] < self.slow_distance)
            ]
            
            if center_obstacles:
                # Obstacle in center path - slow down and steer away
                # Compute lateral offset: centroid x - image_center
                # Average lateral offset of center obstacles
                avg_lateral_offset = np.mean([
                    blob['centroid_x'] - image_center_x 
                    for blob in center_obstacles
                ])
                
                # Map lateral offset to steering command (proportional)
                # Normalize offset to [-1, 1] range (assuming max offset is width/2)
                normalized_offset = avg_lateral_offset / (width / 2)
                normalized_offset = np.clip(normalized_offset, -1.0, 1.0)
                
                # Steering: negative offset (left of center) -> steer right (positive)
                #           positive offset (right of center) -> steer left (negative)
                self.avoidance_steer_direction = -self.steering_gain * normalized_offset
                self.obstacle_avoidance_active = True
                self.obstacle_clearance_time = None
                
                if self.depth_frame_count % 10 == 0:
                    self.get_logger().warn(
                        f"Obstacle in center path! Depth: {min([b['min_depth'] for b in center_obstacles]):.2f}m, "
                        f"Lateral offset: {avg_lateral_offset:.1f}px, Steering: {self.avoidance_steer_direction:.2f}"
                    )
            else:
                # Obstacles present but not blocking center path
                # Could still slow down but don't need aggressive steering
                if self.obstacle_avoidance_active:
                    # Start clearance timer
                    if self.obstacle_clearance_time is None:
                        self.obstacle_clearance_time = time.time()
                    elif time.time() - self.obstacle_clearance_time > self.obstacle_clearance_duration:
                        self.obstacle_avoidance_active = False
                        self.obstacle_clearance_time = None
                        self.avoidance_steer_direction = 0.0
                        self.get_logger().info("Obstacle cleared, resuming normal navigation")
        else:
            # No obstacles detected
            self.emergency_stop_active = False
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
        
        # Store processed data for visualization
        self.last_obstacle_mask = obstacle_mask
        self.last_obstacle_blobs = obstacle_blobs

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
        Enhanced visualization showing ROI, obstacle blobs, and processing results.
        """
        if depth_image is None:
            return
        
        height, width = depth_image.shape
        image_center_x = width // 2
        
        # Draw ROI regions
        # Lower ROI
        lower_start = int(height * (1.0 - self.roi_lower_fraction))
        cv2.line(depth_colormap, (0, lower_start), (width, lower_start), (0, 255, 255), 2)
        cv2.putText(depth_colormap, 'LOWER ROI', (10, lower_start - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Central horizontal band
        center_band_start = int(height * (0.5 - self.roi_center_band_fraction / 2))
        center_band_end = int(height * (0.5 + self.roi_center_band_fraction / 2))
        cv2.line(depth_colormap, (0, center_band_start), (width, center_band_start), (255, 255, 0), 2)
        cv2.line(depth_colormap, (0, center_band_end), (width, center_band_end), (255, 255, 0), 2)
        cv2.putText(depth_colormap, 'CENTER BAND ROI', (10, center_band_start - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Draw center line
        cv2.line(depth_colormap, (image_center_x, 0), (image_center_x, height), (255, 255, 255), 1)
        
        # Draw center ROI boundaries (30% to 70% of width)
        center_roi_start = int(width * 0.3)
        center_roi_end = int(width * 0.7)
        cv2.line(depth_colormap, (center_roi_start, 0), (center_roi_start, height), (0, 255, 0), 1)
        cv2.line(depth_colormap, (center_roi_end, 0), (center_roi_end, height), (0, 255, 0), 1)
        
        # Draw obstacle mask overlay if available
        if self.last_obstacle_mask is not None:
            # Create colored overlay for obstacles
            obstacle_overlay = np.zeros_like(depth_colormap)
            obstacle_overlay[self.last_obstacle_mask] = [0, 0, 255]  # Red for obstacles
            # Blend overlay with colormap (modify in place)
            blended = cv2.addWeighted(depth_colormap, 0.7, obstacle_overlay, 0.3, 0)
            depth_colormap[:] = blended  # Copy result back to original array
        
        # Draw detected obstacle blobs
        if self.last_obstacle_blobs:
            for blob in self.last_obstacle_blobs:
                cx = blob['centroid_x']
                cy = blob['centroid_y']
                depth = blob['min_depth']
                area = blob['area']
                
                # Color based on distance
                if depth < self.emergency_distance:
                    color = (0, 0, 255)  # Red - emergency
                elif depth < self.slow_distance:
                    color = (0, 165, 255)  # Orange - slow down
                else:
                    color = (0, 255, 255)  # Yellow - caution
                
                # Draw centroid
                cv2.circle(depth_colormap, (cx, cy), 5, color, -1)
                cv2.circle(depth_colormap, (cx, cy), 10, color, 2)
                
                # Draw label
                label = f"{depth:.2f}m"
                cv2.putText(depth_colormap, label, (cx + 15, cy), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw distance thresholds
        cv2.putText(depth_colormap, f'Stop: {self.stopping_distance}m', (width - 200, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(depth_colormap, f'Slow: {self.slow_distance}m', (width - 200, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(depth_colormap, f'Emergency: {self.emergency_distance}m', (width - 200, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Highlight if obstacle avoidance is active
        if self.emergency_stop_active:
            overlay = depth_colormap.copy()
            cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 255), 15)  # Thick red border
            blended = cv2.addWeighted(overlay, 0.4, depth_colormap, 0.6, 0)
            depth_colormap[:] = blended  # Copy result back
            cv2.putText(depth_colormap, 'EMERGENCY STOP!', (width//2 - 150, height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        elif self.obstacle_avoidance_active:
            overlay = depth_colormap.copy()
            cv2.rectangle(overlay, (0, 0), (width, height), (0, 165, 255), 10)  # Orange border
            blended = cv2.addWeighted(overlay, 0.3, depth_colormap, 0.7, 0)
            depth_colormap[:] = blended  # Copy result back
            steer_text = f"Steering: {self.avoidance_steer_direction:.2f}"
            cv2.putText(depth_colormap, 'OBSTACLE AVOIDANCE', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            cv2.putText(depth_colormap, steer_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

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
        self.log_message(f"Distance to target: ({distance:.2f})")
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

        # Emergency stop has highest priority
        if self.emergency_stop_active:
            # EMERGENCY STOP - brake immediately
            self.send_brake_command(1.0)
            self.send_accelerator_command(0.0)
            self.send_reverse_command(0.0)
            self.send_steer_command(0.0)
            
            if self.navigation_iterations % 10 == 0:
                self.log_message("EMERGENCY STOP ACTIVE - Braking!")
            
            self.navigation_iterations += 1
            return
        
        # Obstacle avoidance takes priority over normal navigation
        if self.obstacle_avoidance_active:
            # Obstacle avoidance mode - use proportional steering (already calculated based on rock position)
            # avoidance_steer_direction is already proportional (-0.6 to 0.6), use it directly
            actual_steer_command = self.avoidance_steer_direction
            
            # Reduce speed when avoiding obstacles
            # Acceleration
            accel_gain = 2.0
            accel_command = accel_gain * remap_clamp(0.0, 1.0, accel_factor, accel_factor * 0.5, abs(actual_steer_command))
            brake_command = 0.0
            
            self.send_steer_command(actual_steer_command)
            self.send_accelerator_command(accel_command)
            self.send_reverse_command(brake_command)
            self.send_brake_command(0.0)
            
            # Log avoidance action periodically
            if self.navigation_iterations % 10 == 0:
                self.log_message(f"Obstacle avoidance: steering {'LEFT' if actual_steer_command < 0 else 'RIGHT'} ({actual_steer_command:.2f})")
            
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
    
    waypoint_order = [0, 1, 17, 8, 2, 5, 15, 13, 11, 7, 16, 18, 4, 19, 10, 3, 6, 12, 14, 9]
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
