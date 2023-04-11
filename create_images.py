#taken from https://stackoverflow.com/questions/34588464/python-how-to-capture-image-from-webcam-on-click-using-opencv
#will edit to accomodate stereo camera - taking captures for both video streams at once

#camera splice
import numpy as np
import cv2


#want to do: splice, get left and right channels separately, and each time we take a screenshot
#it saves two images, left and right separately in diff folders, but with same numebr

cam = cv2.  VideoCapture(0)
cam.set(cv2.CAP_PROP_FPS, 120)

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
s,orignal = cam.read()
height, width, channels = orignal.shape
print(width)
print(height)
img_counter = 0
while(1):
    s,orignal = cam.read()
    left=orignal[0:height,0:int(width/2)]
    right=orignal[0:height,int(width/2):(width)]
    
    if not s:
        print("failed to grab frame")
        break

    cv2.imshow('left',left)
    cv2.imshow('Right',right)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        L_img_name = "0L_{}.png".format(img_counter)
        R_img_name = "0R_{}.png".format(img_counter)
        cv2.imwrite(L_img_name, left)
        cv2.imwrite(R_img_name, right)
        print("{} written!".format(str(L_img_name)+' & '+str(R_img_name)))
        img_counter += 1

cam.release()
cv2.destroyAllWindows()


