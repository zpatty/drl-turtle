#!/usr/bin/env python3
"""
ROS2 Tracker Node with CORRECTED Grid Search + Stereo Saving + Stitching
Grid search now properly tests template at different positions instead of creating random templates
"""

import sys
import time
import os
import socket
import struct
import pickle
import argparse
import json
import yaml

import cv2
import numpy as np

# ROS2 imports
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

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

parser = argparse.ArgumentParser(description="ROS2 Tracker with Corrected Grid Search")
parser.add_argument("--camera-topic", type=str, default="frames")
parser.add_argument("--message-type", type=str, default="turtlecam",
                    choices=["turtlecam", "compressed", "image"])
parser.add_argument("--use-right-camera", action='store_true')
parser.add_argument("--show-display", action='store_true')
parser.add_argument("--save-data", action='store_true')
parser.add_argument("--output-dir", type=str, default="tracking_data")
parser.add_argument("--server-port", type=int, default=9999)
parser.add_argument("--confidence-threshold", type=float, default=0.7,
                    help="Min confidence before triggering scan")
parser.add_argument("--scan-density", type=int, default=6,
                    help="Grid density for scanning (4=sparse, 6=medium, 8=dense)")
parser.add_argument("--roi-file", type=str, default="saved_roi.json",
                    help="File to save/load ROI selection")
parser.add_argument("--force-select-roi", action='store_true',
                    help="Force ROI selection even if saved file exists")

args = parser.parse_args()

# ============================================================================
# TVM Client
# ============================================================================

class TVMClient:
    def __init__(self, host='localhost', port=9999, timeout=30):
        self.host = host
        self.port = port
        
        print(f"[TVM Client] Connecting to inference server at {host}:{port}...")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(timeout)
        
        max_retries = 5
        for i in range(max_retries):
            try:
                self.sock.connect((host, port))
                print(f"[TVM Client] Connected to inference server")
                break
            except ConnectionRefusedError:
                if i < max_retries - 1:
                    print(f"[TVM Client] Connection refused, retrying in 2s...")
                    time.sleep(2)
                else:
                    raise Exception("Could not connect to TVM inference server")
    
    def send_data(self, data):
        msg = pickle.dumps(data)
        msg = struct.pack('>I', len(msg)) + msg
        self.sock.sendall(msg)
    
    def recv_data(self):
        raw_msglen = self.recvall(4)
        if not raw_msglen:
            return None
        msglen = struct.unpack('>I', raw_msglen)[0]
        data = self.recvall(msglen)
        return pickle.loads(data)
    
    def recvall(self, n):
        data = bytearray()
        while len(data) < n:
            packet = self.sock.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return data
    
    def init(self, frame, init_rect):
        request = {'cmd': 'init', 'frame': frame, 'init_rect': init_rect}
        self.send_data(request)
        return self.recv_data()
    
    def track(self, frame):
        request = {'cmd': 'track', 'frame': frame}
        self.send_data(request)
        return self.recv_data()
    
    def reset(self):
        request = {'cmd': 'reset'}
        self.send_data(request)
        return self.recv_data()
    
    def shutdown(self):
        try:
            request = {'cmd': 'shutdown'}
            self.send_data(request)
            self.recv_data()
        except:
            pass
        finally:
            self.sock.close()

# ============================================================================
# ROS2 Tracker Node with CORRECTED Grid Search
# ============================================================================

class TrackerNode(Node):
    def __init__(self, tvm_client):
        super().__init__('siamrpn_tracker_node')
        
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
            self.get_logger().info(f'Subscribed to {args.camera_topic} (TurtleCam with stitching)')
        elif args.message_type == "compressed":
            self.camera_sub = self.create_subscription(
                CompressedImage, args.camera_topic, self.compressed_callback, qos_profile)
        else:
            self.camera_sub = self.create_subscription(
                Image, args.camera_topic, self.image_callback, qos_profile)
        
        self.tvm_client = tvm_client
        self.br = CvBridge()
        # load stitcher
        self.load_stitching_params()

        # Tracking state
        self.initialized = False
        self.init_frame = None
        self.frame_count = 0
        self.track_lost_count = 0
        self.last_good_bbox = None
        self.last_good_centroid = None
        
        # CRITICAL: Template storage for grid search
        self.template_frame = None  # Stores frame when template was created
        self.template_bbox = None   # Stores bbox where template was extracted
        
        # Bbox size constraints
        self.initial_bbox_size = None
        self.max_bbox_growth = 1.5
        
        # Grid search settings
        self.confidence_threshold = args.confidence_threshold
        self.scan_density = args.scan_density
        self.search_attempts = 0
        self.max_search_attempts = np.inf
        
        # FPS tracking
        self.frame_times = []
        self.max_fps_samples = 30
        
        # ROI selection
        self.roi_selected = False
        self.init_rect = None
        

        # Data logging
        self.save_data = args.save_data
        if self.save_data:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.output_dir = os.path.join(args.output_dir, timestamp)
            os.makedirs(os.path.join(self.output_dir, "detection"), exist_ok=True)
            
            # Create directories for stereo + stitched images
            if args.message_type == "turtlecam":
                os.makedirs(os.path.join(self.output_dir, "left"), exist_ok=True)
                os.makedirs(os.path.join(self.output_dir, "right"), exist_ok=True)
                os.makedirs(os.path.join(self.output_dir, "stitched"), exist_ok=True)
                self.get_logger().info('✅ Stereo + Stitched image saving enabled')
            
            self.csv_path = os.path.join(self.output_dir, "centroids.csv")
            with open(self.csv_path, 'w') as f:
                f.write("frame,timestamp,cx,cy,bbox_x,bbox_y,bbox_w,bbox_h,fps,confidence,tracking_status\n")
            self.get_logger().info(f'Saving data to: {self.output_dir}')
        
        self.get_logger().info(f'Tracker with CORRECTED grid search initialized')
        self.get_logger().info(f'Confidence threshold: {self.confidence_threshold}')
        self.get_logger().info(f'Grid search uses TEMPLATE MATCHING (actually searches for turtle!)')
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
            print(f"yaml paths: {yaml_paths}")
            yaml_path = None
            for path in yaml_paths:
                if os.path.exists(path):
                    yaml_path = path
                    break
            print(f"yaml path: {yaml_path}")
            if yaml_path is None:
                self.get_logger().warn('rig_params.yaml not found, using default stitching parameters')
                self.scale = 1.0
                self.rot_deg = 0.0
                self.tx = 0.0
                self.ty = 0.0
                self.alpha = 0.5
            else:
                print("trying to open yaml path")
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
    
    def interactive_stitch(self, left, right):
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

    def is_bbox_valid(self, bbox):
        """Check if bbox size is reasonable"""
        if self.initial_bbox_size is None:
            return True
        
        bbox_w, bbox_h = bbox[2], bbox[3]
        init_w, init_h = self.initial_bbox_size
        
        width_ratio = bbox_w / init_w
        height_ratio = bbox_h / init_h
        
        if width_ratio > self.max_bbox_growth or height_ratio > self.max_bbox_growth:
            return False
        
        return True
    
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
    
    def stitch_frames_simple(self, left_frame, right_frame):
        """Simple horizontal concatenation of stereo pair"""
        stitched = np.hstack((left_frame, right_frame))
        return stitched
    
    def turtlecam_callback(self, msg):
        try:
            # Decode both cameras
            left_frame = self.br.compressed_imgmsg_to_cv2(msg.data[0], desired_encoding='bgr8')
            right_frame = self.br.compressed_imgmsg_to_cv2(msg.data[1], desired_encoding='bgr8')
            
            # Save individual frames
            if self.save_data:
                self.save_stereo_images(left_frame, right_frame, self.frame_count + 1)
            
            # Stitch for wide-view tracking
            stitched_frame = self.interactive_stitch(left_frame, right_frame)
            
            # Save stitched frame
            # if self.save_data:
            #     stitched_path = os.path.join(self.output_dir, "stitched", 
            #                                 f"frame{self.frame_count + 1:05d}.jpg")
            #     cv2.imwrite(stitched_path, stitched_frame)
            
            # Track on stitched (wider) image
            self.process_frame(stitched_frame)
            
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
    
    def generate_fullscreen_scan_positions(self, frame_shape, last_bbox, grid_density=6):
        """Generate positions to scan entire frame"""
        h, w = frame_shape[:2]
        
        if self.initial_bbox_size is not None:
            box_w, box_h = self.initial_bbox_size
        elif last_bbox is None:
            box_w, box_h = w // 5, h // 5
        else:
            box_w, box_h = last_bbox[2], last_bbox[3]
        
        positions = []
        step_x = w // grid_density
        step_y = h // grid_density
        
        for y in range(0, h - box_h, step_y):
            for x in range(0, w - box_w, step_x):
                positions.append((x, y, box_w, box_h))
        
        return positions
    
    def try_continuous_scan_CORRECTED(self, frame):
        """
        CORRECTED grid search using template matching
        Actually searches for the turtle template at different positions!
        """
        self.get_logger().info('🔍 Starting CORRECTED grid search (template matching)...')
        
        # Check if we have a template
        if self.template_frame is None or self.template_bbox is None:
            self.get_logger().warn('No template available for grid search!')
            return None, 0.0
        
        # Extract turtle template from saved frame
        tx, ty, tw, th = self.template_bbox
        template = self.template_frame[ty:ty+th, tx:tx+tw]
        
        if template.size == 0:
            self.get_logger().warn('Invalid template extracted!')
            return None, 0.0
        
        # Convert to grayscale for faster matching
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Generate scan positions
        scan_positions = self.generate_fullscreen_scan_positions(
            frame.shape, self.template_bbox, grid_density=self.scan_density
        )
        
        self.get_logger().info(f'Testing template at {len(scan_positions)} positions...')
        
        best_bbox = None
        best_confidence = 0.0
        
        h_frame, w_frame = frame.shape[:2]
        
        for i, search_bbox in enumerate(scan_positions):
            try:
                sx, sy, sw, sh = search_bbox
                
                # Extract search region
                search_region = frame_gray[sy:sy+sh, sx:sx+sw]
                
                if search_region.shape[0] > 0 and search_region.shape[1] > 0:
                    # Resize template to match search region size
                    template_resized = cv2.resize(template_gray, (sw, sh))
                    
                    # Template matching (normalized cross-correlation)
                    result = cv2.matchTemplate(search_region, template_resized, cv2.TM_CCOEFF_NORMED)
                    confidence = float(result[0][0])
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_bbox = search_bbox
                        self.get_logger().info(
                            f'  Position {i+1}/{len(scan_positions)}: NEW BEST = {confidence:.3f}'
                        )
                
                # Visual feedback every 5 positions
                if args.show_display and i % 2 == 0:
                    vis_frame = frame.copy()
                    
                    # Draw all grid positions
                    for pos in scan_positions:
                        cv2.rectangle(vis_frame, 
                                    (pos[0], pos[1]),
                                    (pos[0] + pos[2], pos[1] + pos[3]),
                                    (100, 100, 100), 1)
                    
                    # Highlight current position
                    cv2.rectangle(vis_frame, 
                                (sx, sy), (sx+sw, sy+sh),
                                (0, 255, 255), 3)
                    
                    # Progress bar
                    progress = int((i / len(scan_positions)) * w_frame)
                    cv2.rectangle(vis_frame, (0, h_frame-10), (progress, h_frame), (0, 255, 255), -1)
                    
                    # Info text
                    cv2.putText(vis_frame, f"Grid Search: {i+1}/{len(scan_positions)}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.putText(vis_frame, f"Best match: {best_confidence:.3f}", 
                              (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    
                    # Show template in corner
                    template_display = cv2.resize(template, (100, 100))
                    vis_frame[10:110, w_frame-110:w_frame-10] = template_display
                    cv2.putText(vis_frame, "Template", 
                              (w_frame-100, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    
                    cv2.imshow('Tracker', vis_frame)
                    cv2.waitKey(1)
                    
            except Exception as e:
                self.get_logger().error(f'Scan position {i+1} failed: {e}')
                continue
        
        if best_confidence > self.confidence_threshold:
            self.get_logger().info(
                f'✅ Grid search SUCCESS! Found template at confidence: {best_confidence:.3f}'
            )
            return best_bbox, best_confidence
        else:
            self.get_logger().warn(
                f'❌ Grid search failed. Best confidence: {best_confidence:.3f}'
            )
            return None, 0.0
    
    def process_frame(self, frame):
        """Main tracking logic with CORRECTED grid search"""
        self.frame_count += 1
        
        # Initialize tracker on first frame
        if not self.initialized:
            if not self.roi_selected:
                self.init_frame = frame.copy()
                self.select_roi()
                return
            
            try:
                response = self.tvm_client.init(self.init_frame, self.init_rect)
                if response['status'] == 'success':
                    self.initialized = True
                    self.initial_bbox_size = (self.init_rect[2], self.init_rect[3])
                    
                    # IMPORTANT: Save initial template
                    self.template_frame = self.init_frame.copy()
                    self.template_bbox = self.init_rect
                    
                    self.get_logger().info(f'Tracker initialized with ROI: {self.init_rect}')
                    self.get_logger().info(f'✅ Template saved for grid search')
                else:
                    self.get_logger().error(f'Init failed: {response["message"]}')
                    self.roi_selected = False
                    return
            except Exception as e:
                self.get_logger().error(f'TVM server communication failed: {e}')
                return
        
        # Track in subsequent frames
        t0 = time.time()
        try:
            response = self.tvm_client.track(frame)
        except Exception as e:
            self.get_logger().error(f'TVM tracking failed: {e}')
            return
        t1 = time.time()
        
        if response['status'] != 'success':
            self.get_logger().error(f'Tracking error: {response.get("message", "Unknown")}')
            return
        
        outputs = response['outputs']
        inference_time = response['inference_time']
        
        # Update FPS
        frame_time = t1 - t0
        self.frame_times.append(frame_time)
        if len(self.frame_times) > self.max_fps_samples:
            self.frame_times.pop(0)
        avg_fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))
        
        # Extract bbox and confidence
        if 'bbox' in outputs:
            bbox = list(map(int, outputs['bbox']))
            confidence = outputs.get('best_score', 0.0)
            
            # Validate bbox size
            # if not self.is_bbox_valid(bbox):
            #     self.get_logger().warn(f'Rejecting oversized bbox (conf: {confidence:.3f})')
            #     self.track_lost_count += 1
            #     self.centroid_pub.publish(Float32MultiArray(data=[]))
            #     return
            
            # Check if confidence is low - trigger grid search
            if confidence < self.confidence_threshold and self.search_attempts < self.max_search_attempts:
                self.get_logger().warn(
                    f'⚠️  Low confidence ({confidence:.3f}), triggering grid search...'
                )
                
                # Run CORRECTED grid search
                found_bbox, found_conf = self.try_continuous_scan_CORRECTED(frame)
                
                if found_bbox is not None:
                    # Re-initialize tracker with found bbox
                    try:
                        response = self.tvm_client.init(frame, tuple(found_bbox))
                        if response['status'] == 'success':
                            bbox = list(map(int, found_bbox))
                            confidence = found_conf
                            self.search_attempts = 0
                            self.track_lost_count = 0
                            self.get_logger().info('✅ Tracker re-initialized from grid search')
                        else:
                            self.search_attempts += 1
                    except Exception as e:
                        self.get_logger().error(f'Re-initialization failed: {e}')
                        self.search_attempts += 1
                else:
                    self.search_attempts += 1
            else:
                # Good tracking - reset search attempts
                if confidence >= self.confidence_threshold:
                    self.search_attempts = 0
            
            # Update template when tracking is VERY good
            # if confidence >= 0.7:
            #     self.template_frame = frame.copy()
            #     self.template_bbox = tuple(bbox)
            #     self.last_good_bbox = bbox
            
            cx = bbox[0] + bbox[2] / 2.0
            cy = bbox[1] + bbox[3] / 2.0
            
            centroid_msg = Float32MultiArray(data=[cy, cx])
            
            self.track_lost_count = 0
            
            # Save data
            if self.save_data:
                with open(self.csv_path, 'a') as f:
                    f.write(f"{self.frame_count},{time.time()},{cx},{cy},"
                           f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},"
                           f"{avg_fps:.2f},{confidence:.4f},tracking\n")
            
            # Visualization
            if args.show_display or self.save_data:
                cv2.rectangle(frame, (bbox[0], bbox[1]),
                            (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                            (0, 255, 0), 2)
                
                cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
                cv2.line(frame, (int(cx)-15, int(cy)), (int(cx)+15, int(cy)), (0, 0, 255), 2)
                cv2.line(frame, (int(cx), int(cy)-15), (int(cx), int(cy)+15), (0, 0, 255), 2)
                
                info_text = f"FPS: {avg_fps:.1f} | TVM: {1000*inference_time:.1f}ms"
                cv2.putText(frame, info_text, (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                centroid_text = f"Centroid: ({int(cx)}, {int(cy)})"
                cv2.putText(frame, centroid_text, (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                conf_text = f"Confidence: {confidence:.3f}"
                if confidence > 0.9:
                    conf_color = (0, 255, 0)
                    self.centroid_pub.publish(centroid_msg)
                elif confidence > self.confidence_threshold:
                    conf_color = (0, 165, 255)
                    self.centroid_pub.publish(centroid_msg)
                else:
                    conf_color = (0, 0, 255)
                    self.centroid_pub.publish(Float32MultiArray(data=[]))
                
                cv2.putText(frame, conf_text, (10, 90),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, conf_color, 2)
                
                # Show if grid search is active
                if self.search_attempts > 0:
                    search_text = f"Grid searching... (attempt {self.search_attempts})"
                    cv2.putText(frame, search_text, (10, 120),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                
                if self.save_data:
                    frame_path = os.path.join(self.output_dir, "detection", 
                                            f"frame{self.frame_count:05d}.jpg")
                    cv2.imwrite(frame_path, frame)
                
                if args.show_display:
                    cv2.imshow('Tracker', frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('r'):
                        self.reset_tracker()
                    elif key == 27:
                        raise KeyboardInterrupt
                    
                    try:
                        compressed_img = self.br.cv2_to_compressed_imgmsg(frame)
                        self.video_pub.publish(compressed_img)
                    except:
                        pass
            
            if self.frame_count % 30 == 0:
                self.get_logger().info(
                    f'Frame {self.frame_count} | FPS: {avg_fps:.1f} | '
                    f'Centroid: [{int(cy)}, {int(cx)}] | Conf: {confidence:.3f}'
                )
        else:
            # Lost tracking
            self.track_lost_count += 1
            self.centroid_pub.publish(Float32MultiArray(data=[]))
            
            if self.save_data:
                with open(self.csv_path, 'a') as f:
                    f.write(f"{self.frame_count},{time.time()},,,,,,,{avg_fps:.2f},,lost\n")
            
            if self.track_lost_count == 1:
                self.get_logger().warn('Lost tracking!')
    
    def save_roi(self, roi, filename):
        """Save ROI to JSON file"""
        try:
            roi_data = {
                'x': int(roi[0]),
                'y': int(roi[1]),
                'width': int(roi[2]),
                'height': int(roi[3]),
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            with open(filename, 'w') as f:
                json.dump(roi_data, f, indent=2)
            self.get_logger().info(f'ROI saved to {filename}')
            return True
        except Exception as e:
            self.get_logger().error(f'Failed to save ROI: {e}')
            return False
    
    def load_roi(self, filename):
        """Load ROI from JSON file"""
        try:
            if not os.path.exists(filename):
                return None
            
            with open(filename, 'r') as f:
                roi_data = json.load(f)
            
            roi = (
                roi_data['x'],
                roi_data['y'],
                roi_data['width'],
                roi_data['height']
            )
            self.get_logger().info(f'Loaded saved ROI: {roi}')
            return roi
        except Exception as e:
            self.get_logger().error(f'Failed to load ROI: {e}')
            return None
    
    def select_roi(self):
        """Select ROI for tracker initialization"""
        roi_file = args.roi_file
        
        # Try to load saved ROI
        if not args.force_select_roi:
            saved_roi = self.load_roi(roi_file)
            if saved_roi is not None:
                self.init_rect = saved_roi
                self.roi_selected = True
                self.get_logger().info(f'Using saved ROI: {self.init_rect}')
                return
        
        # Manual selection
        if args.show_display:
            self.get_logger().info('Select target ROI in window, then press SPACE or ENTER')
            self.init_rect = cv2.selectROI('Select Target', self.init_frame, False, False)
            cv2.destroyWindow('Select Target')
            
            if self.init_rect[2] > 0 and self.init_rect[3] > 0:
                self.roi_selected = True
                self.get_logger().info(f'ROI selected: {self.init_rect}')
                self.save_roi(self.init_rect, roi_file)
            else:
                self.get_logger().warn('Invalid ROI, waiting for next frame')
        else:
            # Headless mode
            h, w = self.init_frame.shape[:2]
            box_w, box_h = w // 5, h // 5
            self.init_rect = (w//2 - box_w//2, h//2 - box_h//2, box_w, box_h)
            self.roi_selected = True
            self.get_logger().info(f'Headless mode - using center ROI: {self.init_rect}')
            self.save_roi(self.init_rect, roi_file)
    
    def reset_tracker(self):
        """Reset tracker"""
        self.get_logger().info('Resetting tracker...')
        try:
            self.tvm_client.reset()
        except:
            pass
        self.initialized = False
        self.roi_selected = False
        self.init_rect = None
        self.track_lost_count = 0
        self.search_attempts = 0
        self.last_good_bbox = None
        self.template_frame = None
        self.template_bbox = None
        self.initial_bbox_size = None
        self.frame_count = 0
        self.frame_times = []
    
    def cleanup(self):
        """Cleanup resources"""
        print('[INFO] Cleaning up...')
        if self.save_data:
            print(f'[INFO] Data saved to: {self.output_dir}')
        if args.show_display:
            cv2.destroyAllWindows()

def main():
    print("="*60)
    print("ROS2 Tracker with CORRECTED Grid Search + Stereo + Stitching")
    print("="*60)
    print(f"Camera: {args.camera_topic}")
    print(f"Grid Search: Template Matching (actually searches for turtle!)")
    print(f"Confidence Threshold: {args.confidence_threshold}")
    print(f"Scan Density: {args.scan_density}x{args.scan_density}")
    if args.message_type == "turtlecam":
        print(f"Stereo: Enabled (left + right + stitched)")
    print(f"TVM Server: localhost:{args.server_port}")
    print("="*60)
    print()
    
    tvm_client = TVMClient(host='localhost', port=args.server_port)
    
    rclpy.init()
    tracker_node = TrackerNode(tvm_client)
    
    try:
        rclpy.spin(tracker_node)
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")
    finally:
        tracker_node.cleanup()
        tvm_client.shutdown()
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
