import cv2
import pickle
import os

video_dir = r'SampleVideo\video1.mp4'

width, height = 107, 48

file_name = os.path.basename(video_dir).split('.')[0]
folder_name = os.path.dirname(video_dir)

video_capture = cv2.VideoCapture(video_dir)

posList = []


def fileAppend(r):
    r = list(dict.fromkeys(r))
    posList.append(r)
    with open(file_name, 'wb') as f:
        pickle.dump(posList, f)


try:
    with open(file_name, 'rb') as f:
        posList = pickle.load(f)
except:
    posList = []

while video_capture.isOpened():
    ret, img = video_capture.read()
    if ret:
        if posList:
            for pos in posList:
                cv2.rectangle(img, (int(pos[0]), int(pos[1])), (int(pos[0] + pos[2]), int(pos[1] + pos[3])),
                              (0, 0, 255), 2)
        r = cv2.selectROIs("select the area", img, fromCenter=False)  # [Top_X, Top_Y, Bottom_X, Bottom_Y]
        print(r)

        try:
            for i in r:
                fileAppend(i)
        except Exception as e:
            print('Hata : ' + str(e))
            pass
        break

cv2.waitKey(0)
cv2.destroyAllWindows()
