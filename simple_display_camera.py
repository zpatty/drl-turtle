import cv2 as cv
import numpy as np

cap = cv.VideoCapture(2)
cv.namedWindow("main", cv.WINDOW_NORMAL)

while True:
    s,frame = cap.read()

    cv.imshow('image', frame)

    k = cv.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

cap.release()
cv.destroyAllWindows()