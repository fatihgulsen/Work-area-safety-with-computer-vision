from SafetyArea import *

detector = SafetyArea(video_dir='SampleVideo/video2.mp4')
detector.video_detection(Keypoints=True, Mask=True)
