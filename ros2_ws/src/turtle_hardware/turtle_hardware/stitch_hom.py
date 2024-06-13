# import cv2 as cv
# import matplotlib.pyplot as plt
# import numpy as np
# import concurrent.futures
# from sys import exit
# import os
# import random
# from operator import sub

# def transform(src_pts, H):
#     # src = [src_pts 1]
#     src = np.pad(src_pts, [(0, 0), (0, 1)], constant_values=1)
#     # pts = H * src
#     pts = np.dot(H, src.T).T
#     # normalize and throw z=1
#     pts = (pts / pts[:,-1].reshape(-1, 1))[:, 0:2]
#     return pts

# # find the ROI of a transformation result
# def warpRect(rect, H):
#     x, y, w, h = rect
#     corners = [[x, y], [x, y + h - 1], [x + w - 1, y], [x + w - 1, y + h - 1]]
#     extremum = cv.transform(corners, H)
#     minx, miny = np.min(extremum[:,0]), np.min(extremum[:,1])
#     maxx, maxy = np.max(extremum[:,0]), np.max(extremum[:,1])
#     xo = int(np.floor(minx))
#     yo = int(np.floor(miny))
#     wo = int(np.ceil(maxx - minx))
#     ho = int(np.ceil(maxy - miny))
#     outrect = (xo, yo, wo, ho)
#     return outrect
# # homography matrix is translated to fit in the screen
# def coverH(rect, H):
#     # obtain bounding box of the result
#     x, y, _, _ = warpRect(rect, H)
#     # shift amount to the first quadrant
#     xpos = int(-x if x < 0 else 0)
#     ypos = int(-y if y < 0 else 0)
#     # correct the homography matrix so that no point is thrown out
#     T = np.array([[1, 0, xpos], [0, 1, ypos], [0, 0, 1]])
#     H_corr = T.dot(H)
#     return (H_corr, (xpos, ypos))
# # pad image to cover ROI, return the shift amount of origin
# def addBorder(img, rect):
#     x, y, w, h = rect
#     tl = (x, y)    
#     br = (x + w, y + h)
#     top = int(-tl[1] if tl[1] < 0 else 0)
#     bottom = int(br[1] - img.shape[0] if br[1] > img.shape[0] else 0)
#     left = int(-tl[0] if tl[0] < 0 else 0)
#     right = int(br[0] - img.shape[1] if br[0] > img.shape[1] else 0)
#     img = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT, value=[0, 0, 0])
#     orig = (left, top)
#     return img, orig

# def check_limits(pts, size):
#     np.clip(pts[:,0], 0, size[1] - 1, pts[:,0])
#     np.clip(pts[:,1], 0, size[0] - 1, pts[:,1])
#     return pts

# def size2rect(size):
#     return (0, 0, size[1], size[0])

# def warpImage(img, H):
#     # tweak the homography matrix to move the result to the first quadrant
#     H_cover, pos = coverH(size2rect(img.shape), H)
#     # find the bounding box of the output
#     x, y, w, h = warpRect(size2rect(img.shape), H_cover)
#     width, height = x + w, y + h
#     # warp the image using the corrected homography matrix
#     warped = cv.warpPerspective(img, H_corr, (width, height))
#     # make the external boundary solid black, useful for masking
#     warped = np.ascontiguousarray(warped, dtype=np.uint8)
#     gray = cv.cvtColor(warped, cv.COLOR_RGB2GRAY)
#     _, bw = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
#     # https://stackoverflow.com/a/55806272/12447766
#     major = cv.__version__.split('.')[0]
#     if major == '3':
#         _, cnts, _ = cv.findContours(bw, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
#     else:
#         cnts, _ = cv.findContours(bw, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
#     warped = cv.drawContours(warped, cnts, 0, [0, 0, 0], lineType=cv.LINE_4)
#     return (warped, pos)

# # only the non-zero pixels are weighted to the average
# def mean_blend(img1, img2):
#     assert(img1.shape == img2.shape)
#     locs1 = np.where(cv.cvtColor(img1, cv.COLOR_RGB2GRAY) != 0)
#     blended1 = np.copy(img2)
#     blended1[locs1[0], locs1[1]] = img1[locs1[0], locs1[1]]
#     locs2 = np.where(cv.cvtColor(img2, cv.COLOR_RGB2GRAY) != 0)
#     blended2 = np.copy(img1)
#     blended2[locs2[0], locs2[1]] = img2[locs2[0], locs2[1]]
#     blended = cv.addWeighted(blended1, 0.5, blended2, 0.5, 0)
#     return blended
# def warpPano(prevPano, img, H, orig):
#     # correct homography matrix
#     T = np.array([[1, 0, -orig[0]], [0, 1, -orig[1]], [0, 0, 1]])
#     H_corr = H.dot(T)
#     # warp the image and obtain shift amount of origin
#     result, pos = warpImage(prevPano, H_corr)
#     xpos, ypos = pos
#     # zero pad the result
#     rect = (xpos, ypos, img.shape[1], img.shape[0])
#     result, _ = addBorder(result, rect)
#     # mean value blending
#     idx = np.s_[ypos : ypos + img.shape[0], xpos : xpos + img.shape[1]]
#     result[idx] = mean_blend(result[idx], img)
#     # crop extra paddings
#     x, y, w, h = cv.boundingRect(cv.cvtColor(result, cv.COLOR_RGB2GRAY))
#     result = result[y : y + h, x : x + w]
#     # return the resulting image with shift amount
#     return (result, (xpos - x, ypos - y))

# def blend_images(imageA, imageB, H):
#     return warpPano(imageA, imageB, H, (0, 0))

# def cv_blend_images(imageA, imageB, H):
#     # move origin to cover the third quadrant
#     H_corr, pos = coverH(size2rect(imageA.shape), H)
#     xpos, ypos = pos
#     # warp the image and paste the original one
#     result = cv.warpPerspective(imageA, H_corr, (5000, 5000))
#     bottom, right = int(0), int(0)
#     if ypos + imageB.shape[0] > result.shape[0]:
#         bottom = ypos + imageB.shape[0] - result.shape[0]
#     if xpos + imageB.shape[1] > result.shape[1]:
#         right = xpos + imageB.shape[1] - result.shape[1]
#     result = cv.copyMakeBorder(result, 0, bottom, 0, right,
#                                cv.BORDER_CONSTANT, value=[0, 0, 0])
#     # mean value blending
#     idx = np.s_[ypos:ypos+imageB.shape[0], xpos:xpos+imageB.shape[1]]
#     result[idx] = mean_blend(result[idx], imageB)
#     # crop extra paddings
#     x,y,w,h = cv.boundingRect(cv.cvtColor(result, cv.COLOR_RGB2GRAY))
#     result = result[0:y+h,0:x+w]
#     # return the resulting image with shift amount
#     return (result, (xpos, ypos))

# if __name__ == '__main__':
#     right = cv2.imread("right/right 1.png")
#     left = cv2.imread("left/left 1.png")
#     H = 
#     blended, _ = cv_blend_images(right, left, H)
#     cv.destroyAllWindows()

from stitching import Stitcher
from stitching.images import Images

class VideoStitcher(Stitcher):

    def initialize_stitcher(self, **kwargs):
        super().initialize_stitcher(kwargs)
        self.cameras = None
        self.cameras_registered = False
        
    def stitch(self, images, feature_masks=[]):
        self.images = Images.of(
            images, self.medium_megapix, self.low_megapix, self.final_megapix
        )

        if not self.cameras_registered:
            imgs = self.resize_medium_resolution()
            features = self.find_features(imgs, feature_masks)
            matches = self.match_features(features)
            imgs, features, matches = self.subset(imgs, features, matches)
            cameras = self.estimate_camera_parameters(features, matches)
            cameras = self.refine_camera_parameters(features, matches, cameras)
            cameras = self.perform_wave_correction(cameras)
            self.estimate_scale(cameras)
            self.cameras = cameras
            self.cameras_registered = True

        imgs = self.resize_low_resolution()
        imgs, masks, corners, sizes = self.warp_low_resolution(imgs, self.cameras)
        self.prepare_cropper(imgs, masks, corners, sizes)
        imgs, masks, corners, sizes = self.crop_low_resolution(
            imgs, masks, corners, sizes
        )
        self.estimate_exposure_errors(corners, imgs, masks)
        seam_masks = self.find_seam_masks(imgs, corners, masks)

        imgs = self.resize_final_resolution()
        imgs, masks, corners, sizes = self.warp_final_resolution(imgs, self.cameras)
        imgs, masks, corners, sizes = self.crop_final_resolution(
            imgs, masks, corners, sizes
        )
        self.set_masks(masks)
        imgs = self.compensate_exposure_errors(corners, imgs)
        seam_masks = self.resize_seam_masks(seam_masks)

        self.initialize_composition(corners, sizes)
        self.blend_images(imgs, seam_masks, corners)
        return self.create_final_panorama()