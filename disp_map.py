
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
imgL = cv.imread('0L_1.png', cv.IMREAD_GRAYSCALE)
imgR = cv.imread('0R_1.png', cv.IMREAD_GRAYSCALE)
# window_size = 3

#references for figuring out parameters:
# https://stackoverflow.com/questions/22630356/documentation-of-cvstereobmstate-for-disparity-calculation-with-cvstereobm
# https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html 

stereo = cv.StereoBM_create(numDisparities=128, blockSize=95)
#prefiltering parameters: indended to normalize brightness & enhance texture
stereo.setPreFilterSize(5)
#^must be odd & btwn 5 & 255
stereo.setPreFilterCap(20)
#^must be btwn 1 and 63
stereo.setPreFilterType(0)

#stereo correspondence parameters: find matches between camera views
stereo.setMinDisparity(0)
stereo.setTextureThreshold(12)

#post filtering parameters: prevent false matches, help filter at boundaries
stereo.setSpeckleRange(20)
stereo.setSpeckleWindowSize(50)
stereo.setUniquenessRatio(8)

disparity = stereo.compute(imgL,imgR)

plt.imshow(disparity/255,'gray')
plt.show()


#someone wrote a GUI meant to help you tune parameters, here's a class they wrote
#may be useful? but needs some time investment to figure out 
#https://erget.wordpress.com/2014/03/13/building-an-interactive-gui-with-opencv/ 

# class BMTuner(object):

#     """
#     A class for tuning Stereo BM settings.
#     Display a normalized disparity picture from two pictures captured with a
#     ``CalibratedPair`` and allow the user to manually tune the settings for the
#     ``BlockMatcher``.
#     The settable parameters are intelligently read from the ``BlockMatcher``,
#     relying on the ``BlockMatcher`` exposing them as ``parameter_maxima``.
#     """

#     #: Window to show results in
#     window_name = "BM Tuner"

#     def _set_value(self, parameter, new_value):
#         """Try setting new parameter on ``block_matcher`` and update map."""
#         try:
#             self.block_matcher.__setattr__(parameter, new_value)
#         except BadBlockMatcherArgumentError:
#             return
#         self.update_disparity_map()

#     def _initialize_trackbars(self):
#         """
#         Initialize trackbars by discovering ``block_matcher``'s parameters.
#         """
#         for parameter in self.block_matcher.parameter_maxima.keys():
#             maximum = self.block_matcher.parameter_maxima[parameter]
#             if not maximum:
#                 maximum = self.shortest_dimension
#             cv.createTrackbar(parameter, self.window_name,
#                                self.block_matcher.__getattribute__(parameter),
#                                maximum,
#                                partial(self._set_value, parameter))

#     def _save_bm_state(self):
#         """Save current state of ``block_matcher``."""
#         for parameter in self.block_matcher.parameter_maxima.keys():
#             self.bm_settings[parameter].append(
#                                self.block_matcher.__getattribute__(parameter))

#     def __init__(self, block_matcher, calibration, image_pair):
#         """
#         Initialize tuner window and tune given pair.
#         ``block_matcher`` is a ``BlockMatcher``, ``calibration`` is a
#         ``StereoCalibration`` and ``image_pair`` is a rectified image pair.
#         """
#         #: Stereo calibration to find Stereo BM settings for
#         self.calibration = calibration
#         #: (left, right) image pair to find disparity between
#         self.pair = image_pair
#         #: Block matcher to be tuned
#         self.block_matcher = block_matcher
#         #: Shortest dimension of image
#         self.shortest_dimension = min(self.pair[0].shape[:2])
#         #: Settings chosen for ``BlockMatcher``
#         self.bm_settings = {}
#         for parameter in self.block_matcher.parameter_maxima.keys():
#             self.bm_settings[parameter] = []
#         cv.namedWindow(self.window_name)
#         self._initialize_trackbars()
#         self.tune_pair(image_pair)

#     def update_disparity_map(self):
#         """
#         Update disparity map in GUI.
#         The disparity image is normalized to the range 0-255 and then divided by
#         255, because OpenCV multiplies it by 255 when displaying. This is
#         because the pixels are stored as floating points.
#         """
#         disparity = self.block_matcher.get_disparity(self.pair)
#         norm_coeff = 255 / disparity.max()
#         cv.imshow(self.window_name, disparity * norm_coeff / 255)
#         cv.waitKey()

#     def tune_pair(self, pair):
#         """Tune a pair of images."""
#         self._save_bm_state()
#         self.pair = pair
#         self.update_disparity_map()

#     def report_settings(self, parameter):
#         """
#         Report chosen settings for ``parameter`` in ``block_matcher``.
#         ``bm_settings`` is updated to include the latest state before work is
#         begun. This state is removed at the end so that the method has no side
#         effects. All settings are reported except for the first one on record,
#         which is ``block_matcher``'s default setting.
#         """
#         self._save_bm_state()
#         report = []
#         settings_list = self.bm_settings[parameter][1:]
#         unique_values = list(set(settings_list))
#         value_frequency = {}
#         for value in unique_values:
#             value_frequency[settings_list.count(value)] = value
#         frequencies = value_frequency.keys()
#         frequencies.sort(reverse=True)
#         header = "{} value | Selection frequency".format(parameter)
#         left_column_width = len(header[:-21])
#         right_column_width = 21
#         report.append(header)
#         report.append("{}|{}".format("-" * left_column_width,
#                                     "-" * right_column_width))
#         for frequency in frequencies:
#             left_column = str(value_frequency[frequency]).center(
#                                                              left_column_width)
#             right_column = str(frequency).center(right_column_width)
#             report.append("{}|{}".format(left_column, right_column))
#         # Remove newest settings
#         for param in self.block_matcher.parameter_maxima.keys():
#             self.bm_settings[param].pop(-1)
#         return "\n".join(report)