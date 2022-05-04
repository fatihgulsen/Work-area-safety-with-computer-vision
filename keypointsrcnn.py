import torch
import torchvision
import cv2
import time
from PIL import Image
from torchvision.transforms import transforms as transforms
import matplotlib
import numpy as np
import os
import pickle


class KeypointsRCNN:

    def __init__(self, video_dir: str, min_size: int = 500):
        self.model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True,
                                                                            num_keypoints=17, min_size=min_size)
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.video_dir = video_dir
        self.frame_count = 0
        self.total_fps = 0
        self.pTime = 0
        self.model.to(self.device).eval()
        self.edges = [
            (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10),
            (5, 7), (7, 9), (5, 11), (11, 13), (13, 15), (6, 12),
            (12, 14), (14, 16), (5, 6)
        ]

        self.outputs = None
        self.area_list = None
        self.file_name = os.path.basename(self.video_dir).split('.')[0]
        self.folder_name = os.path.dirname(self.video_dir)
        self.__read_pickle_pos(self.file_name)

    def video_detection(self):
        cap = cv2.VideoCapture(self.video_dir)
        if not cap.isOpened():
            print('Error while trying to read video. Please check path again')
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                image, orig_frame = self.__image_trasform(frame)
                orig_frame = self.__draw_area(self.area_list, frame)

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
                fps = 1 / (cTime - self.pTime)
                self.pTime = cTime

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

    def setup_area(self):
        video_capture = cv2.VideoCapture(self.video_dir)
        while video_capture.isOpened():
            ret, img = video_capture.read()
            if ret:
                if self.area_list:
                    for pos in self.area_list:
                        cv2.rectangle(img, (int(pos[0]), int(pos[1])), (int(pos[0] + pos[2]), int(pos[1] + pos[3])),
                                      (0, 0, 255), 2)
                r = cv2.selectROIs("select the area", img, fromCenter=False)  # [Top_X, Top_Y, Bottom_X, Bottom_Y]
                print(r)

                try:
                    for i in r:
                        self.__file_append(i, self.file_name)
                except Exception as e:
                    print('Hata : ' + str(e))
                    pass
                break

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        pass

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
                foot_keypoints_index = [15, 16]
                self.__area_control(self.area_list, keypoints[foot_keypoints_index],image)

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

                    cv2.line(img=image, pt1=(int(keypoints[e, 0][0]), int(keypoints[e, 1][0])),
                             pt2=(int(keypoints[e, 0][1]), int(keypoints[e, 1][1])),
                             color=tuple(rgb), thickness=2, lineType=cv2.LINE_AA)

                cv2.rectangle(image, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])),
                              color=(0, 255, 0),
                              thickness=2)
            else:
                continue
        return image

    def __draw_area(self, area_list, image):
        if area_list:
            for pos in area_list:
                cv2.rectangle(image, (int(pos[0]), int(pos[1])), (int(pos[0] + pos[2]), int(pos[1] + pos[3])),
                              (0, 0, 255), 2)
        return image

    def __file_append(self, pos, file_name: str):
        pos = list(dict.fromkeys(pos))
        self.area_list.append(pos)
        with open(file_name, 'wb') as f:
            pickle.dump(self.area_list, f)

    def __read_pickle_pos(self, file_name: str):
        try:
            with open(file_name, 'rb') as f:
                self.area_list = pickle.load(f)
        except Exception as e:
            self.area_list = []

    def __area_control(self, area_list, pos_list,image):
        # print('Area')
        # print((ax1, ay1, ax2, ay2))
        # print('Pos')
        # print((px1, py1))
        if area_list:
            for area in area_list:
                (ax1, ay1, ax2, ay2) = area
                # print(area)
                if pos_list.all():
                    for pos in pos_list:
                        (px1, py1, dummy) = pos
                        # print(pos)
                        # if (ax2 <= px1 <= ax1) and (ay2 <= py1 <= ay1):
                        #     print(' inside ')
                        #     print(pos)
                        #     print(area)
                        if (ax1 <= px1 <= ax2) and (ay2 <= py1 <= ay1):
                            print(' 2 inside')
                            print(pos)
                            print(area)
                        elif (ax2 <= px1 <= ax1) and (ay1 <= py1 <= ay2):
                            print(' 3 inside')
                            print(pos)
                            print(area)
                        elif (ax1 <= px1 <= ax2) and (ay1 <= py1 <= ay2):
                            print(' 4  inside')
                            print(pos)
                            print(area)
                        else:
                            # print('not')
                            # print(pos)
                            # print(area)
                            pass


        pass
