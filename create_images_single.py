#taken from https://stackoverflow.com/questions/34588464/python-how-to-capture-image-from-webcam-on-click-using-opencv
#edited to accomodate ELP stereo camera which captures for both video streams at once

import numpy as np
import cv2

# The ELP camera displays both cameras in one videocapture stream, so I am splicing the capture stream in half to separate the left and right videos
# Each time we take a screenshot it saves who images, left and right in different folders, but with the same image number

cap0 = cv2.VideoCapture(1)
cap1 = cv2.VideoCapture(2)


img_counter = 0
while(1):
    ret0, left = cap1.read()
    ret1, right = cap0.read()

    cv2.imshow('left',left)
    cv2.imshow('right',right)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        L_img_name = "L_{}.png".format(img_counter)
        R_img_name = "R_{}.png".format(img_counter)
        cv2.imwrite(L_img_name, left)
        cv2.imwrite(R_img_name, right)
        print("written!")
        img_counter += 1

cap1.release()
cap0.release()
cv2.destroyAllWindows()


