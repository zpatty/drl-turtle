#!/usr/bin/env python3
"""
TVM Inference Server with YOLOv8 Detection Fallback
Runs in Python 3.10 TVM environment
Provides both tracking and detection capabilities
"""

import sys
import time
import socket
import pickle
import struct
import argparse
import os

sys.path.append("./pysot/pysot")

import cv2
import torch
import numpy as np
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

torch.set_num_threads(1)

class TVMInferenceServer:
    def __init__(self, config_path, snapshot_path, detector_path=None, port=9999):
        print(f"[TVM Server] Starting inference server on port {port}")
        
        # Change to tracker_models directory for TVM model loading
        original_dir = os.getcwd()
        tracker_models_dir = os.path.join(original_dir, 'tracker_models')
        
        if not os.path.exists(tracker_models_dir):
            raise FileNotFoundError(f"tracker_models directory not found at {tracker_models_dir}")
        
        print(f"[TVM Server] Using models from: {tracker_models_dir}")
        os.chdir(tracker_models_dir)
        
        # Load tracking config
        cfg.merge_from_file(os.path.join(original_dir, config_path))
        cfg.CUDA = False
        
        # Build tracker model in TVM mode
        print("[TVM Server] Building TVM tracker model...")
        model = ModelBuilder(mode='tvm')
        
        # Load tracker weights
        print(f"[TVM Server] Loading tracker weights from {snapshot_path}")
        state = torch.load(os.path.join(original_dir, snapshot_path), map_location='cpu')
        model.load_state_dict(state)
        model.eval()
        
        # Change back to original directory
        os.chdir(original_dir)
        
        # Build tracker
        print("[TVM Server] Building tracker...")
        self.tracker = build_tracker(model)
        
        # Load YOLOv8 detector if provided
        self.detector = None
        if detector_path and os.path.exists(detector_path):
            print(f"[TVM Server] Loading detector from {detector_path}")
            try:
                # Check if it's ONNX or PyTorch model
                if detector_path.endswith('.onnx'):
                    print(f"[TVM Server] Detected ONNX model, using ONNX Runtime")
                    import onnxruntime as ort
                    self.detector = ort.InferenceSession(detector_path)
                    self.detector_type = 'onnx'
                    self.detector_enabled = True
                    print("[TVM Server] ONNX detector loaded successfully")
                else:
                    print(f"[TVM Server] Detected PyTorch model, using Ultralytics")
                    from ultralytics import YOLO
                    self.detector = YOLO(detector_path)
                    self.detector_type = 'pytorch'
                    self.detector_enabled = True
                    print("[TVM Server] PyTorch detector loaded successfully")
            except ImportError as e:
                print(f"[TVM Server] Warning: Could not load detector - missing dependency: {e}")
                print(f"[TVM Server] For ONNX: pip install onnxruntime")
                print(f"[TVM Server] For PyTorch: pip install ultralytics")
                self.detector_enabled = False
            except Exception as e:
                print(f"[TVM Server] Warning: Could not load detector: {e}")
                self.detector_enabled = False
        else:
            print("[TVM Server] No detector provided - tracking only mode")
            self.detector_enabled = False
        
        # Server state
        self.initialized = False
        self.frame_count = 0
        self.detection_mode = False
        self.detector_type = None  # 'onnx' or 'pytorch'
        
        # Socket setup
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('localhost', self.port))
        self.sock.listen(1)
        
        print(f"[TVM Server] Waiting for client connection on port {self.port}...")
        self.conn, self.addr = self.sock.accept()
        print(f"[TVM Server] Client connected from {self.addr}")
    
    def detect_objects(self, frame, conf_threshold=0.25):
        """Run YOLOv8 detection on frame"""
        if not self.detector_enabled:
            return []
        
        try:
            if self.detector_type == 'onnx':
                # ONNX Runtime inference
                return self._detect_onnx(frame, conf_threshold)
            else:
                # PyTorch/Ultralytics inference
                return self._detect_pytorch(frame, conf_threshold)
        except Exception as e:
            print(f"[TVM Server] Detection error: {e}")
            return []
    
    def _detect_pytorch(self, frame, conf_threshold):
        """Detect using PyTorch/Ultralytics YOLO"""
        results = self.detector(frame, conf=conf_threshold, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                bbox = [int(x1), int(y1), int(x2-x1), int(y2-y1)]
                
                detections.append({
                    'bbox': bbox,
                    'confidence': conf,
                    'class': cls,
                    'class_name': result.names[cls]
                })
        
        return detections
    
    def _detect_onnx(self, frame, conf_threshold):
        """Detect using ONNX Runtime"""
        import cv2
        import numpy as np
        
        # Preprocess image for YOLOv8
        # YOLOv8 expects: [1, 3, 640, 640] float32, normalized [0,1]
        img = cv2.resize(frame, (640, 640))
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        
        # Run inference
        input_name = self.detector.get_inputs()[0].name
        outputs = self.detector.run(None, {input_name: img})
        
        # YOLOv8 output: [1, 84, 8400] for detection
        # 84 = 4 (bbox) + 80 (classes)
        # 8400 = number of predictions
        predictions = outputs[0][0]  # Remove batch dimension
        
        # Transpose to [8400, 84]
        predictions = predictions.T
        
        # Filter by confidence
        boxes = predictions[:, :4]  # x, y, w, h
        scores = predictions[:, 4:].max(axis=1)
        class_ids = predictions[:, 4:].argmax(axis=1)
        
        # Filter by threshold
        mask = scores > conf_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]
        
        # Convert from center format to corner format
        # Scale back to original image size
        h_orig, w_orig = frame.shape[:2]
        scale_x = w_orig / 640
        scale_y = h_orig / 640
        
        detections = []
        for box, score, cls in zip(boxes, scores, class_ids):
            x_center, y_center, w, h = box
            
            # Convert to corner format and scale
            x1 = int((x_center - w/2) * scale_x)
            y1 = int((y_center - h/2) * scale_y)
            x2 = int((x_center + w/2) * scale_x)
            y2 = int((y_center + h/2) * scale_y)
            
            # Clip to image bounds
            x1 = max(0, min(x1, w_orig))
            y1 = max(0, min(y1, h_orig))
            x2 = max(0, min(x2, w_orig))
            y2 = max(0, min(y2, h_orig))
            
            bbox = [x1, y1, x2-x1, y2-y1]
            
            detections.append({
                'bbox': bbox,
                'confidence': float(score),
                'class': int(cls),
                'class_name': f'class_{cls}'  # Generic name for ONNX
            })
        
        return detections
    
    def recv_data(self):
        """Receive length-prefixed pickled data"""
        raw_msglen = self.recvall(4)
        if not raw_msglen:
            return None
        msglen = struct.unpack('>I', raw_msglen)[0]
        data = self.recvall(msglen)
        return pickle.loads(data)
    
    def send_data(self, data):
        """Send length-prefixed pickled data"""
        msg = pickle.dumps(data)
        msg = struct.pack('>I', len(msg)) + msg
        self.conn.sendall(msg)
    
    def recvall(self, n):
        """Helper to receive n bytes"""
        data = bytearray()
        while len(data) < n:
            packet = self.conn.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return data
    
    def run(self):
        """Main server loop"""
        print("[TVM Server] Ready for inference requests")
        
        try:
            while True:
                request = self.recv_data()
                if request is None:
                    print("[TVM Server] Client disconnected")
                    break
                
                cmd = request['cmd']
                
                if cmd == 'init':
                    # Initialize tracker
                    frame = request['frame']
                    init_rect = request['init_rect']
                    
                    try:
                        self.tracker.init(frame, init_rect)
                        self.initialized = True
                        self.frame_count = 0
                        self.detection_mode = False
                        response = {'status': 'success', 'message': 'Tracker initialized'}
                        print(f"[TVM Server] Tracker initialized with ROI: {init_rect}")
                    except Exception as e:
                        response = {'status': 'error', 'message': str(e)}
                        print(f"[TVM Server] Init failed: {e}")
                    
                    self.send_data(response)
                
                elif cmd == 'track':
                    # Track in frame
                    frame = request['frame']
                    
                    if not self.initialized:
                        response = {'status': 'error', 'message': 'Tracker not initialized'}
                    else:
                        t0 = time.time()
                        outputs = self.tracker.track(frame)
                        t1 = time.time()
                        
                        self.frame_count += 1
                        
                        response = {
                            'status': 'success',
                            'outputs': outputs,
                            'inference_time': t1 - t0,
                            'frame_count': self.frame_count,
                            'detection_mode': False
                        }
                    
                    self.send_data(response)
                
                elif cmd == 'detect':
                    # Run detection fallback
                    frame = request['frame']
                    conf_threshold = request.get('conf_threshold', 0.25)
                    
                    if not self.detector_enabled:
                        response = {
                            'status': 'error',
                            'message': 'Detector not available'
                        }
                    else:
                        t0 = time.time()
                        detections = self.detect_objects(frame, conf_threshold)
                        t1 = time.time()
                        
                        response = {
                            'status': 'success',
                            'detections': detections,
                            'inference_time': t1 - t0,
                            'num_detections': len(detections)
                        }
                        
                        print(f"[TVM Server] Detected {len(detections)} objects in {1000*(t1-t0):.1f}ms")
                    
                    self.send_data(response)
                
                elif cmd == 'reset':
                    # Reset tracker
                    self.initialized = False
                    self.frame_count = 0
                    self.detection_mode = False
                    response = {'status': 'success', 'message': 'Tracker reset'}
                    self.send_data(response)
                    print("[TVM Server] Tracker reset")
                
                elif cmd == 'shutdown':
                    print("[TVM Server] Shutdown requested")
                    response = {'status': 'success', 'message': 'Shutting down'}
                    self.send_data(response)
                    break
                
                else:
                    response = {'status': 'error', 'message': f'Unknown command: {cmd}'}
                    self.send_data(response)
        
        except KeyboardInterrupt:
            print("\n[TVM Server] Interrupted")
        except Exception as e:
            print(f"[TVM Server] Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        print("[TVM Server] Cleaning up...")
        try:
            self.conn.close()
        except:
            pass
        try:
            self.sock.close()
        except:
            pass
        print("[TVM Server] Shutdown complete")

def main():
    parser = argparse.ArgumentParser(description="TVM Inference Server with Detection Fallback")
    parser.add_argument("--config", type=str, required=True, help="tracker config file")
    parser.add_argument("--snapshot", type=str, required=True, help="tracker checkpoint")
    parser.add_argument("--detector", type=str, default=None, help="YOLOv8 model path (optional)")
    parser.add_argument("--port", type=int, default=9999, help="server port")
    args = parser.parse_args()
    
    print("="*60)
    print("TVM Inference Server (Python 3.10)")
    print("With YOLOv8 Detection Fallback")
    print("="*60)
    print(f"Tracker Config: {args.config}")
    print(f"Tracker Snapshot: {args.snapshot}")
    print(f"Detector: {args.detector if args.detector else 'None (tracking only)'}")
    print(f"Port: {args.port}")
    print("="*60)
    print()
    
    server = TVMInferenceServer(args.config, args.snapshot, args.detector, args.port)
    server.run()

if __name__ == "__main__":
    main()
