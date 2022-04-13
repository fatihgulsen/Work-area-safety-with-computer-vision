import cv2
import matplotlib
import numpy as np
import random
import torch

from coco_names import COCO_INSTANCE_CATEGORY_NAMES as coco_names

COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))

edges = [
    (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10),
    (5, 7), (7, 9), (5, 11), (11, 13), (13, 15), (6, 12),
    (12, 14), (14, 16), (5, 6)
]


def draw_keypoints(outputs, image):
    # the `outputs` is list which in-turn contains the dictionaries
    for i in range(len(outputs[0]['keypoints'])):
        keypoints = outputs[0]['keypoints'][i].cpu().detach().numpy()
        # proceed to draw the lines if the confidence score is above 0.9
        if outputs[0]['scores'][i] > 0.9:
            keypoints = keypoints[:, :].reshape(-1, 3)
            for p in range(keypoints.shape[0]):
                # draw the keypoints
                cv2.circle(image, (int(keypoints[p, 0]), int(keypoints[p, 1])),
                           3, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                # uncomment the following lines if you want to put keypoint number
                cv2.putText(image, f"{p}", (int(keypoints[p, 0]+10), int(keypoints[p, 1]-5)),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            for ie, e in enumerate(edges):
                # get different colors for the edges
                rgb = matplotlib.colors.hsv_to_rgb([
                    ie / float(len(edges)), 1.0, 1.0
                ])
                rgb = rgb * 255
                # join the keypoint pairs to draw the skeletal structure
                try:
                    cv2.line(image, (keypoints[e, 0][0], keypoints[e, 1][0]),
                             (keypoints[e, 0][1], keypoints[e, 1][1]),
                             tuple(rgb), 2, lineType=cv2.LINE_AA)
                except Exception as e:
                    print(e)
                    continue

        else:
            continue
    return image
