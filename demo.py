import cv2
import numpy as np
import picture as pic
import fourierDescriptor as fd
import classify
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.externals import joblib
from functools import reduce
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

file = 'C:/Users/50493/Desktop/like-test.mp4'
out_file = 'C:/Users/50493/Desktop/like-res.mp4'
model_path = "./model/"

N = 15

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # 设置编码方式


def frame_resize(frame, w, h):
    if w > h:
        new_frame = frame[0:h, int((w - h) / 2): w - int((w - h) / 2)]
        res = cv2.resize(new_frame, (300, 300), interpolation=cv2.INTER_CUBIC)
        return res
    else:
        new_frame = frame[int((h - w) / 2): h - int((h - w) / 2), 0:w]
        res = cv2.resize(new_frame, (300, 300), interpolation=cv2.INTER_CUBIC)
        return res

def deom1():
    cp = cv2.VideoCapture(file)
    if not cp.isOpened():
        print("video open error!")

    fps = cp.get(cv2.CAP_PROP_FPS)
    frames = cp.get(cv2.CAP_PROP_FRAME_COUNT)
    width = int(cp.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cp.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_writer = cv2.VideoWriter(filename=out_file, fourcc=fourcc, fps=fps,
                                   frameSize=(300, 300))

    clf = joblib.load(model_path + "svm_efd_" + "train_model.m")

    # print(frames)
    for i in range(int(frames)):
        # for i in range(100):
        ret, frame = cp.read()
        if not ret:
            print("file end")
        reframe = frame_resize(frame, width, height)
        # video_writer.write(reframe)
        # cv2.imwrite('C:/Users/50493/Desktop/reframe.png', reframe)
        roi, res = pic.binaryMask(reframe, 0, 0, 300, 300)
        # cv2.imwrite('C:/Users/50493/Desktop/reframe.png', res)

        img, d = fd.fourierDesciptor(res)
        descirptor_in_use = abs(d)
        temp = descirptor_in_use[1]
        feature = []
        for k in range(1, len(descirptor_in_use)):
            x_record = int(100 * descirptor_in_use[k] / temp)
            feature.append(x_record)

        returnVec = np.zeros((1, N))
        for i in range(N):
            returnVec[0, i] = int(feature[i])
        valTest = clf.predict(returnVec)
        if valTest == 11:
            cv2.putText(roi, "like", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1)
        video_writer.write(roi)
        # print(valTest)


def deom2():
    cameraCapture = cv2.VideoCapture(0)
    cameraCapture.open(0)

    cv2.namedWindow('show')

    fps = cameraCapture.get(cv2.CAP_PROP_FPS)
    frames = cameraCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    width = int(cameraCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cameraCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # video_writer = cv2.VideoWriter(filename=out_file, fourcc=fourcc, fps=fps,
    #                                frameSize=(300, 300))

    clf = joblib.load(model_path + "svm_efd_" + "train_model.m")

    ret, frame = cameraCapture.read()

    while ret:
        cv2.waitKey(1)
        ret, frame = cameraCapture.read()
        reframe = frame_resize(frame, width, height)
        roi, res = pic.binaryMask(reframe, 0, 0, 300, 300)

        img, d = fd.fourierDesciptor(res)
        descirptor_in_use = abs(d)
        temp = descirptor_in_use[1]
        feature = []
        for k in range(1, len(descirptor_in_use)):
            x_record = int(100 * descirptor_in_use[k] / temp)
            feature.append(x_record)

        returnVec = np.zeros((1, N))
        for i in range(N):
            returnVec[0, i] = int(feature[i])
        valTest = clf.predict(returnVec)
        if valTest == 11:
            cv2.putText(roi, "like", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1)
        # video_writer.write(roi)
        cv2.imshow('show', roi)


    cameraCapture.release()


if __name__ == '__main__':
    deom2()
