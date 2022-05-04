from keypointsrcnn import *
import pickle

######################Test

area_list = None


try:
    with open('video1', 'rb') as f:
        area_list = pickle.load(f)
except:
    area_list = []

######################Test

detector = KeypointsRCNN(video_dir='SampleVideo/video2.mp4')
detector.video_detection()
detector.setup_area()


