from keypointsrcnn import *
from maskrcnn import *


class SafetyArea:

    def __init__(self, video_dir: str):
        self.area_list = None
        self.video_dir = video_dir
        self.file_name = os.path.basename(self.video_dir).split('.')[0]
        self.folder_name = os.path.dirname(self.video_dir)
        self.__read_pickle_pos(self.file_name)
        self.edges = [
            (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10),
            (5, 7), (7, 9), (5, 11), (11, 13), (13, 15), (6, 12),
            (12, 14), (14, 16), (5, 6)
        ]
        self.keypointsDetector = KeypointsRCNN()
        self.keypointsOutput = None

        self.maskDetector = MaskRCNN()
        self.masks = None
        self.pred_boxes = None
        self.pred_class = None
        self.frame = None
        pass

    def video_detection(self, Keypoints=True, Mask=True):
        cap = cv2.VideoCapture(self.video_dir)
        if not cap.isOpened():
            print('Error while trying to read video. Please check path again')

        while cap.isOpened():
            ret, self.frame = cap.read()
            if ret:
                start_time = time.time()

                orig_frame = self.__draw_area(self.area_list, self.frame)
                output_image = orig_frame
                if Keypoints:
                    self.keypointsOutput = self.keypointsDetector.get_prediction(self.frame)
                    output_image = self.__draw_keypoints(output_image, self.keypointsOutput)

                if Mask:
                    self.masks, self.pred_boxes, self.pred_class = self.maskDetector.get_prediction(self.frame,0.7)
                    output_image = self.__draw_masks(output_image, self.masks, self.pred_boxes, self.pred_class)
                cv2.imshow('Pose detection frame', output_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            else:
                break
        pass

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

    def __area_control(self, area_list, pos_list, image):
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
                        if (ax2 <= px1 <= ax1) and (ay2 <= py1 <= ay1):
                            print(' inside ')
                            print(pos)
                            print(area)
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

    def __draw_keypoints(self, image, outputs):
        for i in range(len(outputs[0]['keypoints'])):
            if outputs[0]['scores'][i] > 0.9:

                keypoints = outputs[0]['keypoints'][i].cpu().detach().numpy()
                boxes = outputs[0]['boxes'][i].cpu().detach().numpy()

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
                    cv2.line(img=image, pt1=(int(keypoints[e, 0][0]), int(keypoints[e, 1][0])),
                             pt2=(int(keypoints[e, 0][1]), int(keypoints[e, 1][1])),
                             color=tuple(rgb), thickness=2, lineType=cv2.LINE_AA)
                # cv2.rectangle(image, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])),
                #               color=(0, 255, 0),
                #               thickness=2)
            else:
                continue
        return image

    def __draw_masks(self, image, masks, boxes, pred_cls):
        for i in range(len(masks)):
            x0, y0 = boxes[i][0]
            x1, y1 = boxes[i][1]
            rgb_mask = self.__random_colour(masks[i])
            image = cv2.addWeighted(image, 1, rgb_mask, 0.5, 0)
            cv2.rectangle(image, (int(x0), int(y0)), (int(x1), int(y1)), color=(0, 255, 0), thickness=2)
            cv2.putText(image, pred_cls[i], (int(x0), int(y0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                        thickness=1)
        return image

    def __random_colour(self, image):
        colours = [[0, 255, 0], [0, 0, 255], [255, 0, 0],
                   [0, 255, 255], [255, 255, 0], [255, 0, 255],
                   [80, 70, 180],
                   [250, 80, 190], [245, 145, 50], [70, 150, 250],
                   [50, 190, 190]]
        r = np.zeros_like(image).astype(np.uint8)
        g = np.zeros_like(image).astype(np.uint8)
        b = np.zeros_like(image).astype(np.uint8)
        r[image == 1], g[image == 1], b[image == 1] = colours[random.randrange(0, 10)]
        coloured_mask = np.stack([r, g, b], axis=2)
        return coloured_mask
