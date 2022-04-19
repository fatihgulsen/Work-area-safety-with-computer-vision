import torch
import torchvision
import cv2
import time
from PIL import Image
from torchvision.transforms import transforms as transforms
import matplotlib
import numpy as np


class KeypointsRCNN:

    def __init__(self, min_size: int = 500):

        self.model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True,
                                                                            num_keypoints=17, min_size=min_size)
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.frame_count = 0
        self.total_fps = 0
        self.pTime = 0

        self.edges = [
            (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10),
            (5, 7), (7, 9), (5, 11), (11, 13), (13, 15), (6, 12),
            (12, 14), (14, 16), (5, 6)
        ]

        self.outputs = None

    def video_detection(self, video_dir: str):
        cap = cv2.VideoCapture(video_dir)
        if not cap.isOpened():
            print('Error while trying to read video. Please check path again')

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        while cap.isOpened():

            ret, frame = cap.read()
            if ret:
                image, orig_frame = self.__image_trasform(frame)
                start_time = time.time()
                with torch.no_grad():
                    self.outputs = self.model(image)

                end_time = time.time()
                output_image = self.__draw_keypoints(self.outputs, orig_frame)

                fps = 1 / (end_time - start_time)

                self.total_fps += fps

                self.frame_count += 1
                wait_time = max(1, int(fps / 4))

                cTime = time.time()
                fps = 1 / (cTime - pTime)
                pTime = cTime

                cv2.putText(output_image, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                            (255, 0, 0), 3)

                cv2.imshow('Pose detection frame', output_image)

                if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                    break
            else:
                break

        cap.release()

        cv2.destroyAllWindows()

        avg_fps = self.total_fps / self.frame_count
        print(f"Average FPS: {avg_fps:.3f}")

    def __image_trasform(self, image):
        pil_image = Image.fromarray(image).convert('RGB')
        orig_frame = image
        image = self.transform(pil_image)
        image = image.unsqueeze(0).to(self.device)
        return image, orig_frame

    def __draw_keypoints(self, outputs, image):

        for i in range(len(outputs[0]['keypoints'])):

            keypoints = outputs[0]['keypoints'][i].cpu().detach().numpy()

            boxes = outputs[0]['boxes'][i].cpu().detach().numpy()

            if outputs[0]['scores'][i] > 0.9:
                keypoints = keypoints[:, :].reshape(-1, 3)
                for p in range(keypoints.shape[0]):

                    cv2.circle(image, (int(keypoints[p, 0]), int(keypoints[p, 1])),
                               3, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                    cv2.putText(image, f"{p}", (int(keypoints[p, 0] + 10), int(keypoints[p, 1] - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                for ie, e in enumerate(self.edges):

                    rgb = matplotlib.colors.hsv_to_rgb([
                        ie / float(len(self.edges)), 1.0, 1.0
                    ])
                    rgb = rgb * 255

                    try:
                        cv2.line(image, (keypoints[e, 0][0], keypoints[e, 1][0]),
                                 (keypoints[e, 0][1], keypoints[e, 1][1]),
                                 tuple(rgb), 2, lineType=cv2.LINE_AA)
                    except:
                        pass

                cv2.rectangle(image, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])),
                              color=(0, 255, 0),
                              thickness=2)
            else:
                continue
        return image

