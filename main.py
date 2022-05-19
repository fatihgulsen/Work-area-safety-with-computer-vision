from SafetyArea import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--type", type=str, help="Setup or Run")
parser.add_argument("-v", "--video", type=str, help="Video dir")
parser.add_argument("-m", "--mask", type=int, choices=[0, 1], help="Mask on =1 mask off=0")
args = parser.parse_args()


def main():
    detector = SafetyArea(video_dir=args.video, GPU=True)
    if args.type == 'setup':
        detector.setup_area()
    elif args.type == 'run':
        if args.mask == 1:
            detector.video_detection(Keypoints=True, Mask=True)
        elif args.mask == 0:
            detector.video_detection(Keypoints=True, Mask=False)


main()
