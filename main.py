from keypointsrcnn import *
import pickle

####################
detector = KeypointsRCNN(video_dir='SampleVideo/video2.mp4')
detector.video_detection()
detector.setup_area()
#########################

from SafetyArea import *

detector = SafetyArea(video_dir='SampleVideo/video1.mp4')
detector.video_detection(Keypoints=True, Mask=True)
