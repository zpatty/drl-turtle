from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
sys.path.append("/home/tortuga/tortuga_tracker/Yolo-FastestV2/pysot/pysot")
import os
import cv2
import time
import argparse
import time
import torch
import model.detector
import utils.utils
import numpy as np

import time
import os
import argparse

import cv2
import torch
import numpy as np
from glob import glob

from pysot.core.config import cfg as tracker_cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

import utils.loss
import utils.utils
import utils.datasets

class Tracker:
    def __init__(self, use_tracker = 0):
        self.data = '/home/tortuga/tortuga_tracker/Yolo-FastestV2/turtle.data'
        self.detector_weights = '/home/tortuga/tortuga_tracker/Yolo-FastestV2/weights/turtle-30-epoch-0.917089ap-model.pth'
        self.tracker_config = '/home/tortuga/tortuga_tracker/Yolo-FastestV2/pysot/pysot/experiments/siamrpn_alex_dwxcorr_otb/config.yaml'
        self.tracker_snapshot = '/home/tortuga/tortuga_tracker/Yolo-FastestV2/pysot/pysot/experiments/siamrpn_alex_dwxcorr_otb/model.pth'
        self.use_tracker = use_tracker
        #Load label names
        self.LABEL_NAMES = []
        self.cfg = utils.utils.load_datafile(self.data)
        self.cfg["names"] = os.path.expanduser("~") + "/tortuga_tracker/Yolo-FastestV2" + self.cfg["names"]
        with open(self.cfg["names"], 'r') as f:
            for line in f.readlines():
                self.LABEL_NAMES.append(line.strip())

        #Model loading
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detection_model = model.detector.Detector(self.cfg["classes"], self.cfg["anchor_num"], True).to(self.device)
        self.detection_model.load_state_dict(torch.load(self.detector_weights, map_location=self.device))

        #sets the module in eval node
        self.detection_model.eval()

        tracker_cfg.merge_from_file(self.tracker_config)
        tracker_cfg.CUDA = torch.cuda.is_available() and tracker_cfg.CUDA

        tracker_model = ModelBuilder()

        tracker_model.load_state_dict(torch.load(self.tracker_snapshot,
            map_location=lambda storage, loc: storage.cpu()))
        tracker_model.eval().to(self.device)
        
        # build tracker
        self.tracker = build_tracker(tracker_model)

    def preprocess_image(self, ori_img):
        res_img = cv2.resize(ori_img, (self.cfg["width"], self.cfg["height"]), interpolation = cv2.INTER_LINEAR) 
        img = res_img.reshape(1, self.cfg["height"], self.cfg["width"], 3)
        img = torch.from_numpy(img.transpose(0,3, 1, 2))
        img = img.to(self.device).float() / 255.0
        return img
    
    def detect_objects(self, ori_img):
        #ori_img = cv2.imread(opt.img)

        
        img = self.preprocess_image(ori_img)

        start = time.perf_counter()
        preds = self.detection_model(img)
        end = time.perf_counter()
        
        time_took = (end - start) * 1000.
        #print("forward time:%fms"%time_took)

        #Feature map post-processing
        output = utils.utils.handel_preds(preds, self.cfg, self.device)
        output_boxes = utils.utils.non_max_suppression(output, conf_thres = 0.5, iou_thres = 0.5)
        
        return output_boxes
    
    def track(self, frame):
    #init_rect = cv2.selectROI("Object Detection", frame, False, False)


        #print(frame)
        # a=time.time()
        outputs = self.tracker.track(frame)
        # b= time.time()
        #print(b-a)
        if 'polygon' in outputs:
            polygon = np.array(outputs['polygon']).astype(np.int32)
            cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                        True, (0, 255, 0), 3)
            mask = ((outputs['mask'] > self.cfg.TRACK.MASK_THERSHOLD) * 255)
            mask = mask.astype(np.uint8)
            mask = np.stack([mask, mask*255, mask]).transpose(1, 2, 0)
            frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
            return polygon
        else:
            bbox = list(map(int, outputs['bbox']))
            points = np.array([[bbox[0], bbox[1]],[bbox[0]+bbox[2], bbox[1]+bbox[3]]])
            centroid = (points[0] + points[1])/2
            cv2.rectangle(frame, (bbox[0], bbox[1]),
                        (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                        (0, 255, 0), 3)
            return centroid
            #cv2.imshow("Object Detection", frame)
            #cv2.waitKey(40)
    def detect_and_track(self, frame):

    
        frame_count = 0
        total_time = 0
        d = 0


        # Perform object detection on the frame
        start_time = time.time()
        output_boxes = self.detect_objects(frame)
        end_time = time.time()

        # Calculate the time taken for detection in the current frame
        elapsed_time = end_time - start_time
        total_time += elapsed_time
        
        #print(frame_count/total_time)

        # Display the frame
        h, w, _ = frame.shape
        scale_h, scale_w = h / self.cfg["height"], w / self.cfg["width"]
        
        boxes = []
        max_is=-1
        rects = []
        for box in output_boxes[0]:
            
            box = box.tolist()
            obj_score = box[4]
            category = self.LABEL_NAMES[int(box[5])]
            x1, y1 = int(box[0] * scale_w), int(box[1] * scale_h)
            x2, y2 = int(box[2] * scale_w), int(box[3] * scale_h)
            #boxes.append(((x1, y1), (x2, y2), obj_score, category))
            if max_is < obj_score:
                points = np.array([[x1, y1],[x2,y2]])
                centroid = (points[0] + points[1])/2
                max_is = obj_score
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            frame = cv2.putText(frame, '%.2f' % obj_score, (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2)  
            frame = cv2.putText(frame, category, (x1, y1 - 25), 0, 0.7, (0, 255, 0), 2)
            #cv2.imshow("Object Detection", frame)
            if obj_score>0.8:
                #cv2.imshow("Object Detection", frame)
                #cv2.imwrite(f"{d}.png", frame)
                d+=1
                if self.use_tracker:
                    #input_char = input("track?")
                    #cv2.imshow("Object Detection", frame)
                    #if input_char == "y":
                    #  centroid = self.track([x1, y1, x2-x1, y2-y1], frame)
                    rects.append([x1, y1, x2-x1, y2-y1])
        
        # if max_is>-1:
        #     return centroid'
        return rects
        


        # Calculate the average time taken per frame
        avg_time_per_frame = total_time / frame_count
        frames_per_second = 1/avg_time_per_frame
        print("Average fps:", frames_per_second)
    def quick_eval(self):
        batch_size=64
        nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
        val_dataset = utils.datasets.TensorDataset(self.cfg["val"], self.cfg["width"], self.cfg["height"], imgaug = False)
        val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    collate_fn=utils.datasets.collate_fn,
                                                    num_workers=nw,
                                                    pin_memory=True,
                                                    drop_last=False,
                                                    persistent_workers=True
                                                    )
        precision, recall, AP, f1 = utils.utils.evaluation(val_dataloader, self.cfg, self.detection_model, self.device, 0.3)
        print("Precision:%f Recall:%f AP:%f F1:%f"%(precision, recall, AP, f1))

    
