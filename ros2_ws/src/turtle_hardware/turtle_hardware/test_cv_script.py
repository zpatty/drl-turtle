#!/usr/bin/python

import os
import sys
import json
import traceback

# Standard imports
import cv2
import numpy as np
import json 

def parse_cv_params():
    with open('cv_config.json') as config:
        param = json.load(config)
    print(f"[MESSAGE] Config: {param}\n")    
    # Serializing json
    config_params = json.dumps(param, indent=14)
    return param, config_params

def fixHSVRange(val):
    # Normal H,S,V: (0-360,0-100%,0-100%)
    # OpenCV H,S,V: (0-180,0-255 ,0-255)
    return (180 * val[0] / 360, 255 * val[1] / 100, 255 * val[2] / 100)



def main(args=None):
    home_dir = os.path.expanduser("~")

    cv_params, __ = parse_cv_params()
    DIM=(640, 480)

    lower_yellow = np.array(cv_params["lower_yellow"])
    upper_yellow = np.array(cv_params["upper_yellow"])
    cap0 = cv2.VideoCapture(0)
    # cap0.set(3, 1920)
    # cap0.set(4, 1080)
    # cap1 = cv2.VideoCapture(2)

    print("active video feed")

    while(True):
        # ret0, left = cap1.read()
        ret1, right = cap0.read()
        denoise = 15
        blur = cv2.GaussianBlur(right, (5,5), 1)
        # Converting from BGR to HSV color space
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        lab = cv2.cvtColor(right, cv2.COLOR_BGR2LAB)
        bin_y = cv2.inRange(hsv, fixHSVRange(lower_yellow), fixHSVRange(upper_yellow))
        open_kern = np.ones((10,10), dtype=np.uint8)
        bin_y = cv2.morphologyEx(bin_y, cv2.MORPH_OPEN, open_kern, iterations=2)

        rip_y = right.copy()
        rip_y[bin_y==0] = 0
        mark_y = cv2.addWeighted(right, .4, rip_y, .6, 1)

        mask = bin_y #cv2.bitwise_not(bin_y)
        kernel = np.ones((10,10),np.uint8)

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt_s = sorted(cnts, key=cv2.contourArea)

        # Find and draw blocks
        if not (len(cnts) == 0):
            cnt = cnt_s[-1]
            (x,y),radius = cv2.minEnclosingCircle(cnt)
            center = (int(x),int(y))
            radius = int(radius)
            cv2.circle(mask,center,radius,10,2)
            centerMask=cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
            # cv2.imshow('Mask',centerMask)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            turn_thresh = cv_params["turn_thresh"]
            dive_thresh = cv_params["dive_thresh"]
            centroid = center
            if abs(centroid[0] - DIM[0]/2) < turn_thresh and abs(centroid[1] - DIM[1]/2) < dive_thresh:
                # output straight primitive
                print("go straight...\n")
            elif centroid[0] > DIM[0] - (DIM[0]/2 - turn_thresh):
                # turn right
                print("turn right...\n")
            elif centroid[0] < (DIM[0]/2 - turn_thresh):
                # turn left
                print("turn left...\n")
            elif centroid[1] > DIM[1] - (DIM[1]/2 - dive_thresh):
                # dive
                print("dive...\n")
            elif centroid[1] < (DIM[1]/2 - dive_thresh): 
                # surface
                print("surface...\n")
            else:
                # dwell
                print("dwell...\n")
        else:
            print("no detection")
            # centerMask=cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
            # cv2.imshow('Mask',centerMask)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break


    	
    cv2.destroyAllWindows()
    print("closing")

if __name__ == '__main__':
    main()
