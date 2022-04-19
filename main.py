import torch
import torchvision
import cv2
import argparse
import utils
import time
from PIL import Image
from torchvision.transforms import transforms as transforms
import matplotlib

edges = [
    (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10),
    (5, 7), (7, 9), (5, 11), (11, 13), (13, 15), (6, 12),
    (12, 14), (14, 16), (5, 6)
]


def draw_keypoints_and_boxes(outputs, image):
    # the `outputs` is list which in-turn contains the dictionary
    for i in range(len(outputs[0]['keypoints'])):
        # get the detected keypoints
        keypoints = outputs[0]['keypoints'][i].cpu().detach().numpy()
        # get the detected bounding boxes
        boxes = outputs[0]['boxes'][i].cpu().detach().numpy()
        # proceed to draw the lines and bounding boxes
        if outputs[0]['scores'][i] > 0.9:  # proceed if confidence is above 0.9
            keypoints = keypoints[:, :].reshape(-1, 3)
            for p in range(keypoints.shape[0]):
                # draw the keypoints
                cv2.circle(image, (int(keypoints[p, 0]), int(keypoints[p, 1])),
                           3, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(image, f"{p}", (int(keypoints[p, 0] + 10), int(keypoints[p, 1] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            # draw the lines joining the keypoints
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
                except:
                    pass
            # draw the bounding boxes around the objects
            cv2.rectangle(image, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])),
                          color=(0, 255, 0),
                          thickness=2)
        else:
            continue
    return image


# transform to convert the image to tensor
transform = transforms.Compose([
    transforms.ToTensor()
])
# initialize the model
model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True,
                                                               num_keypoints=17)
# set the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# load the modele on to the computation device and set to eval mode
model.to(device).eval()

cap = cv2.VideoCapture('SampleVideo/video2.mp4')
if not cap.isOpened():
    print('Error while trying to read video. Please check path again')
# get the video frames' width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
# set the save path

# define codec and create VideoWriter object

frame_count = 0  # to count total frames
total_fps = 0  # to get the final frames per second
pTime = 0
# read until end of video
while cap.isOpened():
    # capture each frame of the video
    ret, frame = cap.read()
    if ret:

        pil_image = Image.fromarray(frame).convert('RGB')
        orig_frame = frame
        # transform the image
        image = transform(pil_image)
        # add a batch dimension
        image = image.unsqueeze(0).to(device)
        # get the start time
        start_time = time.time()
        with torch.no_grad():
            outputs = model(image)
        # get the end time
        end_time = time.time()
        output_image = draw_keypoints_and_boxes(outputs, orig_frame)
        # get the fps
        fps = 1 / (end_time - start_time)
        # add fps to total fps
        total_fps += fps
        # increment frame count
        frame_count += 1
        wait_time = max(1, int(fps / 4))

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(output_image, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)

        cv2.imshow('Pose detection frame', output_image)
        # press `q` to exit
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break
    else:
        break

# release VideoCapture()
cap.release()
# close all frames and video windows
cv2.destroyAllWindows()
# calculate and print the average FPS
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")
