from SafetyArea import *

detector = SafetyArea(video_dir='SampleVideo/securityvideo1.mp4', GPU=True)
# detector.setup_area()
detector.video_detection(Keypoints=True, Mask=False)
