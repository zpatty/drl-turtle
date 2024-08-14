import numpy as np
import cv2 as cv


radial_dist = [-0.224417104508741,	0.0423129098859732,	-0.00340467161640536]
radial_dist = [0.285864117264439,	-3.19165537934906,	9.35714487821002]
tan_dist = [0., 0.]
dist = np.array(radial_dist[0:2] + tan_dist + [radial_dist[2]])
mtx = np.array([[793.095982230189,	0,	888.918539593734],
			[0,	706.097338009412,	285.609231524379],
			[0,	0,	1]])

mtx = np.array([[738.519435557609,	0,	556.588428078619],
				[0,	784.251357607748,	190.583152186242],
				[0,	0,	1]])
img = cv.imread('left/left 2.png')
h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
 
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imshow('calibresult.png', dst)
cv.imshow('original.png', img)
cv.waitKey()
cv.destroyAllWindows()