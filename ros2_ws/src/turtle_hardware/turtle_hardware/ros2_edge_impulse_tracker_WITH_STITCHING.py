#!/usr/bin/env python3
"""
ROS2 Edge Impulse Tracker with Stereo Stitching
Uses ImageImpulseRunner with auto_studio_settings + stereo stitching for wide view
"""

import sys
import time
import os
import argparse
import yaml

import cv2
import numpy as np

# ROS2 imports
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

# Edge Impulse with correct SDK
from edge_impulse_linux.image import ImageImpulseRunner

# Custom turtle messages
try:
    from turtle_interfaces.msg import TurtleCam
    HAS_TURTLE_INTERFACES = True
except ImportError:
    HAS_TURTLE_INTERFACES = False
    print("[WARNING] turtle_interfaces not found, TurtleCam support disabled")

# ============================================================================
# Configuration
# ============================================================================

parser = argparse.ArgumentParser(description="ROS2 Edge Impulse Tracker with Stereo Stitching")
parser.add_argument("--model", type=str, required=True, help="Path to .eim model file")
parser.add_argument("--camera-topic", type=str, default="frames")
parser.add_argument("--message-type", type=str, default="turtlecam",
                    choices=["turtlecam", "compressed", "image"])
parser.add_argument("--use-stitching", action='store_true',
                    help="Use stereo stitching for wide view (TurtleCam only)")
parser.add_argument("--target-class", type=str, default="turtle")
parser.add_argument("--confidence-threshold", type=float, default=0.5)
parser.add_argument("--show-display", action='store_true')
parser.add_argument("--save-data", action='store_true')
parser.add_argument("--output-dir", type=str, default="tracking_data")

args = parser.parse_args()

# ============================================================================
# ROS2 Tracker Node
# ============================================================================

class EdgeImpulseTrackerNode(Node):
    def __init__(self, model_path):
        super().__init__('edge_impulse_tracker_node')
        
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=2
        )
        
        # Publishers
        self.centroid_pub = self.create_publisher(Float32MultiArray, 'centroids', qos_profile)
        
        if args.show_display:
            self.video_pub = self.create_publisher(CompressedImage, 'video_detect', qos_profile)
        
        # Subscribers
        if args.message_type == "turtlecam":
            if not HAS_TURTLE_INTERFACES:
                raise ImportError("TurtleCam requested but turtle_interfaces not available")
            self.camera_sub = self.create_subscription(
                TurtleCam, args.camera_topic, self.turtlecam_callback, qos_profile)
            self.get_logger().info(f'Subscribed to {args.camera_topic} (TurtleCam)')
            
            # Load stitching parameters if using stitching
            if args.use_stitching:
                self.load_stitching_params()
                self.get_logger().info('Stereo stitching enabled')
        elif args.message_type == "compressed":
            self.camera_sub = self.create_subscription(
                CompressedImage, args.camera_topic, self.compressed_callback, qos_profile)
        else:
            self.camera_sub = self.create_subscription(
                Image, args.camera_topic, self.image_callback, qos_profile)
        
        self.br = CvBridge()
        
        # Tracking state
        self.frame_count = 0
        self.last_detection = None
        self.track_lost_count = 0
        
        # FPS tracking
        self.frame_times = []
        self.max_fps_samples = 30
        
        # Load Edge Impulse model with CORRECT SDK
        self.get_logger().info(f'Loading Edge Impulse model: {model_path}')
        
        # Ensure ./ prefix
        if not model_path.startswith('/') and not model_path.startswith('./'):
            model_path = './' + model_path
        
        self.runner = ImageImpulseRunner(model_path)
        model_info = self.runner.init()
        
        self.model_info = model_info
        self.labels = model_info['model_parameters']['labels']
        self.target_class = args.target_class
        self.confidence_threshold = args.confidence_threshold
        
        self.get_logger().info(f"✅ Model loaded successfully")
        self.get_logger().info(f"   Project: {model_info['project']['owner']} / {model_info['project']['name']}")
        self.get_logger().info(f"   Labels: {self.labels}")
        self.get_logger().info(f"   Target class: {self.target_class}")
        self.get_logger().info(f"   Confidence threshold: {self.confidence_threshold}")
        
        # Data logging
        self.save_data = args.save_data
        if self.save_data:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.output_dir = os.path.join(args.output_dir, timestamp)
            os.makedirs(os.path.join(self.output_dir, "detection"), exist_ok=True)
            
            # Create directories for stereo if using TurtleCam
            if args.message_type == "turtlecam":
                os.makedirs(os.path.join(self.output_dir, "left"), exist_ok=True)
                os.makedirs(os.path.join(self.output_dir, "right"), exist_ok=True)
                if args.use_stitching:
                    os.makedirs(os.path.join(self.output_dir, "stitched"), exist_ok=True)
            
            self.csv_path = os.path.join(self.output_dir, "centroids.csv")
            with open(self.csv_path, 'w') as f:
                f.write("frame,timestamp,cx,cy,bbox_x,bbox_y,bbox_w,bbox_h,fps,confidence,tracking_status\n")
            self.get_logger().info(f'Saving data to: {self.output_dir}')
        
        self.get_logger().info('Edge Impulse tracker initialized')
        self.get_logger().info('Waiting for first frame...')
    
    def load_stitching_params(self):
        """Load stereo stitching parameters from YAML file"""
        try:
            # Look for rig_params.yaml in current directory or parent
            yaml_paths = [
                'rig_params.yaml',
                '../rig_params.yaml',
                os.path.expanduser('~/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/rig_params.yaml')
            ]
            
            yaml_path = None
            for path in yaml_paths:
                if os.path.exists(path):
                    yaml_path = path
                    break
            
            if yaml_path is None:
                self.get_logger().warn('rig_params.yaml not found, using default stitching parameters')
                self.scale = 1.0
                self.rot_deg = 0.0
                self.tx = 0.0
                self.ty = 0.0
                self.alpha = 0.5
            else:
                with open(yaml_path, 'r') as f:
                    params = yaml.safe_load(f)
                
                self.scale = float(params.get('scale', 1.0))
                self.rot_deg = float(params.get('rot_deg', 0.0))
                self.tx = float(params.get('tx', 0.0))
                self.ty = float(params.get('ty', 0.0))
                self.alpha = float(params.get('alpha', 0.5))
                
                self.get_logger().info(f'Loaded stitching params from {yaml_path}')
                self.get_logger().info(f'  scale={self.scale}, rot={self.rot_deg}°, tx={self.tx}, ty={self.ty}')
            
            # Build affine transformation matrix
            self.dx = self.tx
            self.dy = self.ty
            self.theta = np.deg2rad(self.rot_deg)
            self.affine_matrix = np.array([
                [self.scale * np.cos(self.theta), -self.scale * np.sin(self.theta), self.dx],
                [self.scale * np.sin(self.theta), self.scale * np.cos(self.theta), self.dy]
            ])
            
        except Exception as e:
            self.get_logger().error(f'Failed to load stitching params: {e}')
            self.use_stitching = False
    
    def stitch_frames(self, left, right):
        """Stitch left and right stereo images using affine transformation"""
        h, w = left.shape[:2]

        corners = np.array([
            [0, 0],
            [0, h],
            [w, 0],
            [w, h]
        ], dtype=np.float32)

        transformed_corners = cv2.transform(np.array([corners]), self.affine_matrix)[0]
        all_corners = np.vstack((corners, transformed_corners))

        [xmin, ymin] = np.floor(all_corners.min(axis=0)).astype(int)
        [xmax, ymax] = np.ceil(all_corners.max(axis=0)).astype(int)
        output_size = (xmax - xmin, ymax - ymin)
        offset = np.array([-xmin, -ymin])

        affine_with_offset = self.affine_matrix.copy()
        affine_with_offset[:, 2] += offset

        transformed_right = cv2.warpAffine(right, affine_with_offset, output_size)

        canvas = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
        canvas[offset[1]:offset[1] + h, offset[0]:offset[0] + w] = left

        stitched = np.maximum(canvas, transformed_right)

        return stitched
    
    def save_stereo_images(self, left_frame, right_frame, frame_num):
        """Save both left and right camera images"""
        if not self.save_data:
            return
        
        try:
            left_path = os.path.join(self.output_dir, "left", f"frame{frame_num:05d}.jpg")
            right_path = os.path.join(self.output_dir, "right", f"frame{frame_num:05d}.jpg")
            
            cv2.imwrite(left_path, left_frame)
            cv2.imwrite(right_path, right_frame)
        except Exception as e:
            self.get_logger().error(f'Failed to save stereo images: {e}')
    def run_inference(self, frame):
        """Run Edge Impulse inference - FOMO centroid-based detection"""
        try:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            t0 = time.time()
            features, cropped = self.runner.get_features_from_image_auto_studio_settings(img_rgb)
            res = self.runner.classify(features)
            t1 = time.time()
            
            inference_time = t1 - t0
            detections = []
            
            if 'bounding_boxes' in res['result']:
                h_orig, w_orig = frame.shape[:2]
                h_cropped, w_cropped = cropped.shape[:2]
                
                # Calculate letterboxing parameters
                scale = min(w_cropped / w_orig, h_cropped / h_orig)
                scaled_w = int(w_orig * scale)
                scaled_h = int(h_orig * scale)
                pad_x = (w_cropped - scaled_w) // 2
                pad_y = (h_cropped - scaled_h) // 2
                
                for bb in res['result']['bounding_boxes']:
                    if bb['label'] != self.target_class:
                        continue
                    if bb['value'] < self.confidence_threshold:
                        continue
                    
                    # FOMO: Calculate centroid from bbox (this is the primary output!)
                    cx_letterbox = bb['x'] + bb['width'] / 2.0
                    cy_letterbox = bb['y'] + bb['height'] / 2.0
                    
                    # Transform centroid from letterbox to original coordinates
                    cx_unpadded = cx_letterbox - pad_x
                    cy_unpadded = cy_letterbox - pad_y
                    cx_original = cx_unpadded / scale
                    cy_original = cy_unpadded / scale
                    
                    # For visualization, create a reasonable bbox around centroid
                    # (since FOMO's bbox is just grid cells, not object extent)
                    bbox_size = 30  # Arbitrary visualization size in original space
                    x_visual = int(cx_original - bbox_size/2)
                    y_visual = int(cy_original - bbox_size/2)
                    
                    detections.append({
                        'centroid': [cx_original, cy_original],  # PRIMARY OUTPUT
                        'bbox': [x_visual, y_visual, bbox_size, bbox_size],  # Just for visualization
                        'confidence': bb['value'],
                        'label': bb['label']
                    })
            
            return detections, inference_time
            
        except Exception as e:
            self.get_logger().error(f'Inference failed: {e}')
            return [], 0.0
    # def run_inference(self, frame):
    #     """Run Edge Impulse inference using CORRECT SDK"""
    #     try:
    #         # Convert BGR to RGB
    #         img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
    #         t0 = time.time()
    #         features, cropped = self.runner.get_features_from_image_auto_studio_settings(img_rgb)
    #         # cv2.imwrite(f'cropped_debug_{t0}.jpg', cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
    #         h_cropped, w_cropped = cropped.shape[:2]
    #         self.get_logger().info(f'Cropped image size: {w_cropped}×{h_cropped}')
    #         res = self.runner.classify(features)
    #         t1 = time.time()
            
    #         inference_time = t1 - t0
    #         detections = []
            
    #         if 'bounding_boxes' in res['result']:
    #             h_orig, w_orig = frame.shape[:2]  # 482, 937
    #             h_cropped, w_cropped = cropped.shape[:2]  # 482, 482
                
    #             # For 'fit-longest' mode:
    #             # Scale factor = target / longest_dimension
    #             scale = min(w_cropped / w_orig, h_cropped / h_orig)  # 482/937 = 0.514
                
    #             # Scaled dimensions (before padding)
    #             scaled_w = int(w_orig * scale)  # 937 * 0.514 = 482
    #             scaled_h = int(h_orig * scale)  # 482 * 0.514 = 247
                
    #             # Padding offsets (image is centered)
    #             pad_x = (w_cropped - scaled_w) // 2  # (482-482)/2 = 0
    #             pad_y = (h_cropped - scaled_h) // 2  # (482-247)/2 = 117
                
    #             self.get_logger().info(f'DEBUG: scale={scale:.3f}, scaled={scaled_w}×{scaled_h}, pad=({pad_x},{pad_y})')
                
    #             for bb in res['result']['bounding_boxes']:
    #                 print(f"Full bbox data: {bb}")

    #                 if bb['label'] != self.target_class:
    #                     continue
    #                 if bb['value'] < self.confidence_threshold:
    #                     continue
                    
    #                 # Bboxes are in 482×482 letterboxed space
    #                 # Step 1: Remove padding offset
    #                 x_unpadded = bb['x'] - pad_x  # No horizontal padding for this case
    #                 y_unpadded = bb['y'] - pad_y  # Remove 117px vertical padding
    #                 w_unpadded = bb['width']
    #                 h_unpadded = bb['height']
                    
    #                 # Step 2: Scale back to original dimensions
    #                 x = int(x_unpadded / scale)
    #                 y = int(y_unpadded / scale)
    #                 w = int(w_unpadded / scale)
    #                 h = int(h_unpadded / scale)
                    
    #                 # Clip to image bounds
    #                 x = max(0, min(x, w_orig - 1))
    #                 y = max(0, min(y, h_orig - 1))
    #                 w = min(w, w_orig - x)
    #                 h = min(h, h_orig - y)
                    
    #                 self.get_logger().info(f'DEBUG: bbox in letterbox: ({bb["x"]},{bb["y"]},{bb["width"]},{bb["height"]})')
    #                 self.get_logger().info(f'DEBUG: bbox in original: ({x},{y},{w},{h})')
                    
    #                 detections.append({
    #                     'bbox': [x, y, w, h],
    #                     'confidence': bb['value'],
    #                     'label': bb['label']
    #                 })
            
    #         return detections, inference_time
            
    #     except Exception as e:
    #         self.get_logger().error(f'Inference failed: {e}')
    #         return [], 0.0    
    
    # def run_inference(self, frame):
    #     """Run Edge Impulse inference using CORRECT SDK"""
    #     try:
    #         # Convert BGR to RGB (OpenCV uses BGR, Edge Impulse uses RGB)
    #         img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
    #         # Use SDK's automatic preprocessing (matches Studio exactly!)
    #         t0 = time.time()
    #         features, cropped = self.runner.get_features_from_image_auto_studio_settings(img_rgb)
            
    #         # Run inference
    #         res = self.runner.classify(features)
    #         t1 = time.time()
            
    #         inference_time = t1 - t0
            
    #         # Extract bounding boxes
    #         detections = []
            
    #         if 'bounding_boxes' in res['result']:
    #             h_orig, w_orig = frame.shape[:2]  # 482, 937
    #             h_cropped, w_cropped = cropped.shape[:2]  # 482, 482
                
    #             # Calculate letterboxing parameters
    #             # For "Fit longest axis": scale = min(target/width, target/height)
    #             scale = min(w_cropped / w_orig, h_cropped / h_orig)  # 0.514
                
    #             # Calculate scaled size (before padding)
    #             scaled_w = int(w_orig * scale)  # 482
    #             scaled_h = int(h_orig * scale)  # 247
                
    #             # Calculate padding offsets (to center the image)
    #             pad_x = (w_cropped - scaled_w) // 2  # 0
    #             pad_y = (h_cropped - scaled_h) // 2  # 117
                
    #             for bb in res['result']['bounding_boxes']:
    #                 # Only include target class
    #                 if bb['label'] != self.target_class:
    #                     continue
                    
    #                 # Only include if above confidence threshold
    #                 if bb['value'] < self.confidence_threshold:
    #                     continue
                    
    #                 # Bboxes are in letterboxed 482×482 space
    #                 # Step 1: Remove padding offset
    #                 x_unpadded = bb['x'] - pad_x
    #                 y_unpadded = bb['y'] - pad_y
                    
    #                 # Step 2: Scale back to original size
    #                 x = int(x_unpadded / scale)
    #                 y = int(y_unpadded / scale)
    #                 w = int(bb['width'] / scale)
    #                 h = int(bb['height'] / scale)
                    
    #                 # Clip to image bounds
    #                 x = max(0, min(x, w_orig))
    #                 y = max(0, min(y, h_orig))
    #                 w = min(w, w_orig - x)
    #                 h = min(h, h_orig - y)
                    
    #                 detections.append({
    #                     'bbox': [x, y, w, h],
    #                     'confidence': bb['value'],
    #                     'label': bb['label']
    #                 })
            
    #         return detections, inference_time
            
    #     except Exception as e:
    #         self.get_logger().error(f'Inference failed: {e}')
    #         return [], 0.0

    # def run_inference(self, frame):
    #     """Run Edge Impulse inference using CORRECT SDK"""
    #     try:
    #         # Convert BGR to RGB (OpenCV uses BGR, Edge Impulse uses RGB)
    #         img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
    #         # Use SDK's automatic preprocessing (matches Studio exactly!)
    #         t0 = time.time()
    #         features, cropped = self.runner.get_features_from_image_auto_studio_settings(img_rgb)
            
    #         # Run inference
    #         res = self.runner.classify(features)
    #         t1 = time.time()
            
    #         inference_time = t1 - t0
            
    #         # Extract bounding boxes
    #         detections = []
            
    #         if 'bounding_boxes' in res['result']:
    #             h_orig, w_orig = frame.shape[:2]
    #             h_cropped, w_cropped = cropped.shape[:2]
                
    #             for bb in res['result']['bounding_boxes']:
    #                 # Only include target class
    #                 if bb['label'] != self.target_class:
    #                     continue
                    
    #                 # Only include if above confidence threshold
    #                 if bb['value'] < self.confidence_threshold:
    #                     continue
                    
    #                 # Bboxes are in cropped image coordinates
    #                 # Scale back to original frame size
    #                 x = int(bb['x'] * w_orig / w_cropped)
    #                 y = int(bb['y'] * h_orig / h_cropped)
    #                 w = int(bb['width'] * w_orig / w_cropped)
    #                 h = int(bb['height'] * h_orig / h_cropped)
    #                 detections.append({
    #                     'bbox': [x, y, w, h],
    #                     'confidence': bb['value'],
    #                     'label': bb['label']
    #                 })
            
    #         return detections, inference_time
            
    #     except Exception as e:
    #         self.get_logger().error(f'Inference failed: {e}')
    #         return [], 0.0
    
    def turtlecam_callback(self, msg):
        try:
            # Decode both cameras
            left_frame = self.br.compressed_imgmsg_to_cv2(msg.data[0], desired_encoding='bgr8')
            right_frame = self.br.compressed_imgmsg_to_cv2(msg.data[1], desired_encoding='bgr8')
            
            # Save individual frames if logging
            if self.save_data:
                self.save_stereo_images(left_frame, right_frame, self.frame_count + 1)
            
            # Use stitched frame if enabled, otherwise use left camera
            if args.use_stitching:
                frame = self.stitch_frames(left_frame, right_frame)
                
                # Save stitched frame if logging
                if self.save_data:
                    stitched_path = os.path.join(self.output_dir, "stitched", 
                                                f"frame{self.frame_count + 1:05d}.jpg")
                    cv2.imwrite(stitched_path, frame)
            else:
                frame = left_frame
            
            self.process_frame(frame)
            
        except Exception as e:
            self.get_logger().error(f'TurtleCam processing failed: {e}')
    
    def compressed_callback(self, msg):
        try:
            frame = self.br.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.process_frame(frame)
        except Exception as e:
            self.get_logger().error(f'Compressed image conversion failed: {e}')
    
    def image_callback(self, msg):
        try:
            frame = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.process_frame(frame)
        except Exception as e:
            self.get_logger().error(f'Image conversion failed: {e}')
    
    def process_frame(self, frame):
        """Main tracking logic"""
        self.frame_count += 1
        
        # Run inference
        t0 = time.time()
        detections, inference_time = self.run_inference(frame)
        t1 = time.time()
        
        # Update FPS
        frame_time = t1 - t0
        self.frame_times.append(frame_time)
        if len(self.frame_times) > self.max_fps_samples:
            self.frame_times.pop(0)
        avg_fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))
        
        # Process detections
        if len(detections) > 0:
            # Use highest confidence detection
            best_detection = max(detections, key=lambda d: d['confidence'])
            
            bbox = best_detection['bbox']
            conf = best_detection['confidence']
            
            # Calculate centroid
            # cx = bbox[0] + bbox[2] / 2.0
            # cy = bbox[1] + bbox[3] / 2.0
            cx, cy = best_detection['centroid']

            
            # Publish centroid
            centroid_msg = Float32MultiArray(data=[cy, cx])
            self.centroid_pub.publish(centroid_msg)
            
            self.last_detection = best_detection
            self.track_lost_count = 0
            tracking_status = "tracking"
            bbox = best_detection['bbox']
    
            # Draw crosshair at CENTROID (the real detection)
            cv2.circle(frame, (int(cx), int(cy)), 8, (0, 0, 255), -1)
            cv2.line(frame, (int(cx)-20, int(cy)), (int(cx)+20, int(cy)), (0, 0, 255), 3)
            cv2.line(frame, (int(cx), int(cy)-20), (int(cx), int(cy)+20), (0, 0, 255), 3)
            
            # Draw approximate bbox (just for reference, not meaningful in FOMO)
            cv2.rectangle(frame, (bbox[0], bbox[1]),
                        (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                        (0, 255, 0), 2)
            # Save data
            if self.save_data:
                with open(self.csv_path, 'a') as f:
                    f.write(f"{self.frame_count},{time.time()},{cx},{cy},"
                           f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},"
                           f"{avg_fps:.2f},{conf:.4f},{tracking_status}\n")
            
            if self.frame_count % 30 == 0:
                self.get_logger().info(
                    f'Frame {self.frame_count} | FPS: {avg_fps:.1f} | '
                    f'Centroid: [{int(cy)}, {int(cx)}] | Conf: {conf:.3f}'
                )
        
        else:
            # No detections
            self.track_lost_count += 1
            tracking_status = "lost"
            bbox = None
            conf = 0.0
            cx, cy = 0, 0
            
            # Publish empty centroid
            self.centroid_pub.publish(Float32MultiArray(data=[]))
            
            if self.save_data:
                with open(self.csv_path, 'a') as f:
                    f.write(f"{self.frame_count},{time.time()},,,,,,,{avg_fps:.2f},,{tracking_status}\n")
            
            if self.track_lost_count == 1:
                self.get_logger().warn('Lost tracking of turtle!')
        
        # ALWAYS run visualization (even when lost)
        if args.show_display or self.save_data:
            # Draw bounding box if we have a detection
            if bbox is not None:
                cv2.rectangle(frame, (bbox[0], bbox[1]),
                            (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                            (0, 255, 0), 2)
                
                # Draw centroid crosshair
                cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
                cv2.line(frame, (int(cx)-15, int(cy)), (int(cx)+15, int(cy)), (0, 0, 255), 2)
                cv2.line(frame, (int(cx), int(cy)-15), (int(cx), int(cy)+15), (0, 0, 255), 2)
            
            # Info text - ALWAYS show
            info_text = f"FPS: {avg_fps:.1f} | Inference: {1000*inference_time:.1f}ms"
            cv2.putText(frame, info_text, (10, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Status text - show tracking or lost
            if tracking_status == "tracking":
                centroid_text = f"Centroid: ({int(cx)}, {int(cy)})"
                cv2.putText(frame, centroid_text, (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                conf_text = f"Confidence: {conf:.3f}"
                conf_color = (0, 255, 0) if conf > 0.9 else (0, 165, 255)
                cv2.putText(frame, conf_text, (10, 90),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, conf_color, 2)
            else:
                status_text = f"LOST TRACKING ({self.track_lost_count} frames)"
                cv2.putText(frame, status_text, (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Show stitching status
            if args.message_type == "turtlecam" and args.use_stitching:
                stitch_text = "Stitched View"
                cv2.putText(frame, stitch_text, (10, 120),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            if self.save_data:
                frame_path = os.path.join(self.output_dir, "detection", 
                                        f"frame{self.frame_count:05d}.jpg")
                cv2.imwrite(frame_path, frame)
            
            if args.show_display:
                cv2.imshow('Tracker', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    raise KeyboardInterrupt
                
                try:
                    compressed_img = self.br.cv2_to_compressed_imgmsg(frame)
                    self.video_pub.publish(compressed_img)
                except:
                    pass
    
    def cleanup(self):
        """Cleanup resources"""
        self.get_logger().info('Cleaning up...')
        if self.save_data:
            self.get_logger().info(f'Data saved to: {self.output_dir}')
        if args.show_display:
            cv2.destroyAllWindows()
        try:
            self.runner.stop()
        except:
            pass

def main():
    print("="*60)
    print("ROS2 Edge Impulse Tracker with Stereo Stitching")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Camera: {args.camera_topic}")
    print(f"Target: {args.target_class}")
    print(f"Confidence: {args.confidence_threshold}")
    if args.message_type == "turtlecam" and args.use_stitching:
        print(f"Stitching: ENABLED (wide view)")
    print("="*60)
    print()
    
    rclpy.init()
    tracker_node = EdgeImpulseTrackerNode(args.model)
    
    try:
        rclpy.spin(tracker_node)
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")
    finally:
        tracker_node.cleanup()
        try:
            tracker_node.destroy_node()
        except:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except:
            pass

if __name__ == "__main__":
    main()