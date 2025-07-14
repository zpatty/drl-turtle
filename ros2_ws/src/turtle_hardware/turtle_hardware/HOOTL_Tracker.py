# --------------------------------------------------------
# TAn
# Licensed under The MIT License
# Written by Alaa Maalouf (alaam@mit.edu)
# --------------------------------------------------------
import sys
import glob
#from PyQt5.QtCore import QLoggingCategory  # or PyQt6.QtCore for PyQt6, etc.
#import warnings
#warnings.filterwarnings("ignore")
# Disable specific Qt warnings
#QLoggingCategory.setFilterRules("*.debug=false\n*.warning=false")

import cv2
import torch

cv2.namedWindow('Choose_object1')#solving a bug
cv2.destroyAllWindows()
import os
from torchvision.transforms.functional import to_tensor
from PIL import Image
import numpy as np

#Tracker imports
from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model

#Segmentor imports
# sys.path.append("segment-anything")
sys.path.append(os.path.expanduser("~") + "/work/segment-anything")
from segment_anything import sam_model_registry, SamPredictor


#from tapnet.torch import tapir_model
#from tapnet.utils import transforms
#from tapnet.utils import viz_utils

#from DRONE.drone_controller import *
from VIDEO.video import *

from collections import OrderedDict
import copy
import threading
import time
from scipy.signal import butter, filtfilt

import asyncio
import argparse
import matplotlib
import gc 
import queue
from matplotlib.colors import LinearSegmentedColormap


# python track_anything.py --height 400 --width 800 --video /dev/video0  --detection_mode click  --re_detection_mode click --plot_visualization --save_images_to outputs/here --is_stream 1
# parser = argparse.ArgumentParser(description='PyTorch + mavsdk -- zero shot detection, tracking, and drone control')

# parser.add_argument('--use_filter', action='store_true',  default =False, help='use_filter')
# parser.add_argument('--plot_visualizations', action='store_true', default =False, help='plot_visualizations')
# parser.add_argument('--height', default=-1, type=int, help='desired_height resulution')
# parser.add_argument('--width', default=-1, type=int, help='desired_width resulution')
# parser.add_argument('--video', default='video/whales.mp4', help='The path to the video file')
# parser.add_argument('--save_images_to', default=False, help='The path to save all semgentation/tracking frames')

# parser.add_argument('--detection_mode', default = "box", help='')
# parser.add_argument('--re_detection_mode', default = "box", help='')
# #parser.add_argument('--num_of_points_to_track', default = 3,  type=int, help='')


# parser.add_argument('--is_stream', default = 0, type=float, help='realtime_or_not')
# #parser.add_argument('--streaming', default = 0, type=int, help='indicator')
# parser.add_argument('--wait_key', default=1, type=int, help='cv waitkey')

# parser.add_argument('--sam_model_type', default = "vit_h")
# parser.add_argument('--sam_model_path', default = "segment-anything/models/sam_vit_h_4b8939.pth")
# parser.add_argument('--num_of_clicks_for_detection', default=3, type = float,  help='')

# args = parser.parse_args()
import matplotlib as plt
cmap = plt.cm.get_cmap('jet')

class TrackAny:
    def __init__(self, args):
        self.mission_counter = 0
        self.frame_idx = 0
        self.clicks_for_retrack = None
        self.state_for_retrack = 0
        self.points = []
        self.labels = []
        self.state = 0
        self.p1, self.p2 = None, None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.backends.cudnn.benchmark = True
        self.args = args
        self.cfg = vars(args)
        self.cfg['device'] = self.device
        self.button = ""
        
        # Setup Tracker Model
        print("Init tracker...")
        # Obtain the Cutie model with default parameters -- skipping hydra configuration
        self.tracker = self.setup_cutie_tracker()
    
        print("Init Segmentor...")
        self.segmentor = self.setup_sam_segmentor()

        #print("Init point to point tracker...")
        #tapir_points_tracker = setup_tapir_points_tracker(cfg)

        print("Init video...")
        # self.video = self.setup_video(self.cfg)
        
        #print("Init Drone...")
        #drone = setup_drone(cfg)

        print("Init directories to store outputs...")
        self.setup_dirs_to_store_data()
    
        # return device, cfg, cutie_tracker, sam_segmentor

    
    def multiclass_vis(self, class_labels, img_to_viz, num_of_labels, np_used = False,alpha = 0.5):
        _overlay = img_to_viz.astype(float) / 255.0
        if np_used:
            viz = cmap(class_labels/num_of_labels)[..., :3]
        else:
            class_labels = class_labels.detach().cpu().numpy().astype(float)
            viz = cmap((class_labels/num_of_labels))[..., :3]
        _overlay =  alpha * viz + (1-alpha) * _overlay 
        s_overlay = cv2.cvtColor(np.float32(_overlay), cv2.COLOR_BGR2RGB)  

        return _overlay

    def bool_mask_to_integer(self, mask):
        mask_obj = mask[0]
        img = np.zeros((mask_obj.shape[0], mask_obj.shape[1]))
        img[mask_obj] = 1
        return img

        

    #todo
    def get_siammask_tracker(self, siam_cfg, device):

        from custom import Custom
        
        siammask = Custom(anchors=siam_cfg['anchors'])
        if self.args.siam_tracker_model:
            assert isfile(self.args.siam_tracker_model), 'Please download {} first.'.format(self.args.siam_tracker_model)
            siammask = load_pretrain(siammask, self.args.siam_tracker_model)
        siammask.eval().to(device)

        return siammask


                
    def plot_and_save_if_neded(self, image_to_plot, stage_and_task, count, multiply = 1, plot = True):

        if self.cfg['plot_visualizations'] and plot: 
            cv2.imshow(stage_and_task, image_to_plot)
            cv2.waitKey(self.cfg['wait_key'])
        if self.cfg['save_images_to']:
            file_name = "{}/{}/{}_{}.jpg".format(self.cfg['save_images_to'],stage_and_task,self.mission_counter ,count)
            #if os.path.exists(filename):
            cv2.imwrite(file_name,image_to_plot*multiply)
        
        


    def drop_spot_on_click(self, event, x ,y, flags, userdata):


        if event == cv2.EVENT_RBUTTONDOWN:
            print(f"[DEBUG] drop_spot_on_click() Right Click Detected")
            self.state_for_retrack+=1
            self.add = 1
            self.clicks_for_retrack.append([x, y])
            print(f"[CLICK DETECTED] clicks for retrack: {self.clicks_for_retrack}\n")
        if self.state_for_retrack > 1 and self.add:
            # self.clicks_for_retrack.append([x, y])
            self.add = 0
        #print(clicks_for_drop)
    def drop_spot_on_box(self, event, x ,y, flags, userdata):
        # Left click
        if event == cv2.EVENT_RBUTTONDOWN:
            print(f"[DEBUG] drop_spot_on_box() Right Click Detected")

            self.state_for_retrack+=1
            self.add = 1
        if self.state_for_retrack > 2 and self.add:
            if event == cv2.EVENT_RBUTTONDOWN or event == cv2.EVENT_RBUTTONUP:
                self.clicks_for_retrack.append([x, y])
            if event == cv2.EVENT_LBUTTONDOWN:
                print(f"[RESETTING clicks for retrack]")
                self.clicks_for_retrack = []
                self.state_for_retrack = 0
                self.add = 0

    def set_tracker_visualization(self, name = 'Stream_tracking'):
        if self.cfg['plot_visualizations'] or self.cfg['re_detection_mode'] in ['box', 'click']:
            cv2.namedWindow(name)
            if self.cfg['re_detection_mode'] == "click":
                cv2.setMouseCallback(name, self.drop_spot_on_click) 
            else:
                cv2.setMouseCallback(name, self.drop_spot_on_box) 

    def should_finish(self):
        if self.button == "stop":
            print(" Should Finish")
            self.points = []
            self.labels = []
            self.mission_counter = 0
            self.frame_idx = 0
            self.clicks_for_retrack = None
            self.state_for_retrack = 0
            self.points = []
            self.labels = []
            self.state = 0
            self.p1, self.p2 = None, None
            self.button = ""
            return True
        return False

    def handle_plots_while_tracking(self, frame, np_mask, frame_idx):
        vis_masks = self.multiclass_vis(np_mask, frame, np.max(np_mask) + 1, np_used = True)
        # add the two points in orange and red to the vis
        #vis_masks = plot_points_on_frame(vis_masks, plot_points, [True]*len(plot_points), cfg)
        self.plot_and_save_if_neded(frame, "Stream_tracking", frame_idx)
        self.plot_and_save_if_neded(vis_masks, 'Tracker-result',frame_idx,multiply = 255)
        
    def fix_mask_and_objects_for_tracking(self, mask, track_single_object = True):
        mask = mask.astype(int)
        if track_single_object:
            mask[mask!=1] = 0 

        objects = np.unique(np.array(mask))  
        objects = objects[objects != 0].tolist()
    
        mask = torch.from_numpy(np.array(mask)).cuda()
        return mask, objects

    def should_retrack(self, clicks_for_retrack):
        print(f"[SHOULD RETRACK CHECK] detection mode: {self.cfg['re_detection_mode']}, clicks for retrack: {clicks_for_retrack}")
        if self.cfg['re_detection_mode'] == "click" and len(clicks_for_retrack) >= self.cfg['num_of_clicks_for_detection']:
            return True
        if self.cfg['re_detection_mode'] == "box" and len(clicks_for_retrack) >= 2:
            return True
        return False

    @torch.inference_mode()
    @torch.cuda.amp.autocast()
    def track_object_with_cutie(self, mask, frame, track_single_object = True):
        print(f"----[FRAME {self.frame_idx}]----\n")
        if self.frame_idx == 0:
            self.set_tracker_visualization()

            mask, objects = self.fix_mask_and_objects_for_tracking(mask, track_single_object=track_single_object)
            # print("here")
            timing = 0
            avg_time_took = 0

        with torch.cuda.amp.autocast():
            t = time.time()
            # torch.cuda.empty_cache()
            # gc.collect()
            
            tensor_frame = to_tensor(frame).cuda().float()
            ##############################################
            if self.frame_idx == 0:
                self.tracker.delete_objects([1])
                output_prob = self.tracker.step(tensor_frame, mask, objects=objects)
            else:
                output_prob = self.tracker.step(tensor_frame)
            ###############################################
            mask = self.tracker.output_prob_to_mask(output_prob)
            time_for_tracker = time.time() - t
            np_mask = mask.numpy(force=True).astype(float)
            #if track_single_object: mask[mask!=1] = 0 
            ###############################################
            mean_point = self.get_mean_point(np_mask)
            print(f"[DEBUG] mean_point: {mean_point}, np_mask shape: {np_mask.shape}")
            #np.save(f"outputs/{frame_idx}", np_mask)
            #if mean_point is None: return "FAILED", None
            self.handle_plots_while_tracking(frame, np_mask, self.frame_idx)
            if self.should_finish(): return "Success", [], None, None
            if self.should_retrack(self.clicks_for_retrack): return "retrack", [], None, self.clicks_for_retrack

            ##############################################
            #plot_points = compute_drone_action_while_tracking(mean_point, np_mask, cfg, drone)
            ##############################################
            t2=time.time()
            timeforplot = time.time() - t2
            ##############################################
            ##############################################
            ##############################################
            self.frame_idx += 1
            # frame = self.read_one_frame(video)
            avg_time_took = time.time() - t# +avg_time_took*frame_idx)/(frame_idx+1)
            #print(time.time() - t)
            print("processed frame {}, avr_fps for pipiline {}, {}".format(self.frame_idx, 1/avg_time_took, avg_time_took))#,end='\r')
            # print("processed frame {}, avr_fps for tracker  {}, {}".format(self.frame_idx, 1/time_for_tracker,time_for_tracker))#,end='\r')
            #print("processed frame {}, avr_fps for plots {}, {}".format(frame_idx, 1/timeforplot, timeforplot))#,end='\r')
            
            if mean_point is None:
                return "Failed", [], None, None
            return "Tracking", mean_point, mask, None

        
    def get_mean_point(self, pred_mask, bounding_shape = None):
        
        
        if not pred_mask is None:
            object_indx = (pred_mask == 1).nonzero()
            if object_indx[0].shape[0] == 0:
                return None ## restart mission
        
        mean_point = [int(object_indx[0].mean()),  int(object_indx[1].mean())]
        
        return mean_point


    def setup_cutie_tracker(self):
        print("get dddd")
        cutie = get_default_model()
        # Typically, use one InferenceCore per video
        cutie_processor = InferenceCore(cutie, cfg=cutie.cfg)
        # the processor matches the shorter edge of the input to this size
        # you might want to experiment with different sizes, -1 keeps the original size
        cutie_processor.max_internal_size = min(self.cfg['height'], self.cfg['width'])
        return cutie_processor
    
    def setup_sam_segmentor(self):
        # Setup 0-shot segmentor
        sam = sam_model_registry[self.cfg['sam_model_type']](checkpoint=self.cfg['sam_model_path'])
        sam.to(device=self.cfg['device'])
        sam_segmentor = SamPredictor(sam)
        return sam_segmentor
    
    def setup_video(self):
        if os.path.isdir(self.cfg["video"]):
            print("Making video from images in directory {}".format(self.cfg["video"]))
            video = create_video_from_images(self.cfg)
        elif os.path.exists(self.cfg["video"]) and not self.cfg['is_stream']:
            print("Reading video  {}".format(self.cfg["video"]))
            video = cv2.VideoCapture(self.cfg["video"]) 
        else:
            print("Using stream from {}".format(self.cfg["video"]))
            video = ThreadedCamera(self.cfg["video"], fps = 0)
        #input()
        return video

    def setup_dirs_to_store_data(self):
        if self.cfg['save_images_to']:
            self.create_dir_if_doesnt_exists(self.cfg['save_images_to'])
            for directory_to_create in ['Tracker-result', 'Stream_tracking']:
                self.create_dir_if_doesnt_exists(os.path.join(self.cfg['save_images_to'],directory_to_create))
        


    def create_dir_if_doesnt_exists(self, dir_to_create):
        if not os.path.exists(dir_to_create): os.mkdir(dir_to_create)



    # Called every time a mouse event happen
    def on_mouse(self, event, x, y, flags, userdata):
        # Left click
        if event == cv2.EVENT_LBUTTONDOWN:
            # Select first point
                self.p1 = (x,y)
                self.state += 1
        elif event == cv2.EVENT_LBUTTONUP:
            # Select second point
            if self.state == 1:
                self.p2 = (x,y)
                self.state += 1
        # Right click (erase current ROI)
        if event == cv2.EVENT_RBUTTONUP:
            print(f"[DEBUG] on mouse Right Click Detected")
            self.p1, self.p2 = None, None
            self.state = 0

    def click_on_object(self, event, x, y, flags, userdata):
        # Left click
        if event == cv2.EVENT_LBUTTONDOWN:
            # Select first point
            self.points.append([x,y])
            self.labels.append(1)
            self.state += 1
        # Right click (erase current ROI)
        if event == cv2.EVENT_RBUTTONDOWN:
            print(f"[DEBUG] click_on_object() Right Click Detected")
            self.points.append([x,y])
            self.labels.append(0)
            self.state += 1

    # Register the mouse callback

    def detect_by_click(self, frame, clicks = None):
        # global state, points, labels
        # self.points = []
        # self.labels = []
        # self.state = 0
        
        cv2.namedWindow('Choose_object')#, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Choose_object', self.click_on_object)    
        cv2.imshow('Choose_object', frame)
        
        key = cv2.waitKey(self.cfg['wait_key'])
        if key == 27: exit(9)
        if self.clicks_for_retrack is not None and self.clicks_for_retrack != []:
            self.state = self.cfg['num_of_clicks_for_detection']
            self.points = self.clicks_for_retrack
            self.labels = np.ones(len(self.clicks_for_retrack))
        # elif self.button == "retrack":
        #     self.state = self.cfg['num_of_clicks_for_detection']
        #     self.points = [np.floor(np.shape(frame)[1]/2), np.floor(np.shape(frame)[0]/2)]*2
        #     self.labels = np.array([1,1])
        #     self.button = "track"

        if self.state >= self.cfg['num_of_clicks_for_detection']:
            self.segmentor.set_image(frame)
            input_point = np.array(self.points)
            input_label = np.array(self.labels)

            masks, scores, logits = self.segmentor.predict(
                                            point_coords=input_point,
                                            point_labels=input_label,
                                            box = None,
                                            multimask_output=False,
                                        )
            return None, masks, frame 
        
        if self.button == "track":
            self.segmentor.set_image(frame)
            point = [np.floor(np.shape(frame)[1]/2), np.floor(np.shape(frame)[0]/2)]
            p1 = [np.floor(np.shape(frame)[1]/2) + np.floor(np.shape(frame)[1]/10), np.floor(np.shape(frame)[0]/2) + np.floor(np.shape(frame)[1]/10)]
            p2 = [np.floor(np.shape(frame)[1]/2) - np.floor(np.shape(frame)[1]/10), np.floor(np.shape(frame)[0]/2) - np.floor(np.shape(frame)[1]/10)]
            
            input_point = np.array([point]*2)
            input_label = np.array([1, 1])

            masks, scores, logits = self.segmentor.predict(
                                            point_coords=input_point,
                                            point_labels=input_label,
                                            box = None,
                                            multimask_output=False,
                                        )
            input_box = np.array([p2[0], p2[1], p1[0], p1[1]])
            print(input_box)
            masks, _, _ = self.segmentor.predict(
                            point_coords=None,
                            point_labels=None,
                            box=input_box[None, :],
                            multimask_output=False,
                            )
            
            return None, masks, frame 
        
        return None, None, frame
    
    def detect_by_box(self, frame, clicks = None): 
        self.p1, self.p2 = None, None
        self.state = 0
        cv2.namedWindow('Choose_object')
        cv2.setMouseCallback('Choose_object', self.on_mouse)
        # Our ROI, defined by two points
        
        # If a ROI is selected, draw it
        if state > 1:
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 5)
        
        cv2.cvtColor(np.float32(frame), cv2.COLOR_BGR2RGB)
        cv2.imshow('Choose_object', frame)
        # Let OpenCV manage window events
        key = cv2.waitKey(self.cfg['wait_key'])
        # If ESCAPE key pressed, stop
        # if key == 27: video.release()
        
        if clicks is not None:
            state = 2
            p1 = clicks[0]
            p2 = clicks[1]
        if state > 1:
        
            self.segmentor.set_image(frame)
            input_box = np.array([p1[0], p1[1], p2[0], p2[1]])
            masks, _, _ = self.segmentor.predict(
                            point_coords=None,
                            point_labels=None,
                            box=input_box[None, :],
                            multimask_output=False,
                            )
            
            return input_box, masks, frame
    
            #drone_action_wrapper_while_detecting(vehicle,cfg)

    def detect_object(self, frame, clicks = None):
        # print("aplying {} detection...".format(self.cfg['detection_mode']))
        # frame = cv2.resize(frame, (self.cfg['width'],self.cfg['height']))
        if self.cfg['detection_mode'] == "auto":
            print("Not supported yet ...")
            pass
        else:
            if self.button == "retrack":
                print(f"[DEBUG] Retracking based on user command with {clicks}")
                bounding_boxes, masks, saved_frame = self.detect_by_click(frame, clicks=clicks )
            elif self.cfg['detection_mode'] == 'click':
                print(f"[DEBUG] Detecting by click with {clicks}")
                bounding_boxes, masks, saved_frame = self.detect_by_click(frame, clicks=clicks )
            else:
                bounding_boxes, masks, saved_frame = self.detect_by_box(frame, clicks=clicks)
            # print("here")
            if  masks is not None:
                masks_of_sam = self.bool_mask_to_integer(masks)
                vis_masks = self.multiclass_vis(masks_of_sam, saved_frame, 2, np_used = True)
                self.plot_and_save_if_neded(vis_masks, 'Choose_object',0)
            
            else:
                masks_of_sam = None 
        return bounding_boxes, masks_of_sam, saved_frame
            
        

    