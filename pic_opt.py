import cv2
import os

path = 'C:/Users/50493/Desktop/test1/'
outpath = './' + 'images-like' + '/'

if __name__ == '__main__':
    fileList = os.listdir(path)
    print(len(fileList))
    for i in range(len(fileList)):
        print(path + fileList[i])
        pic = cv2.imread(path + fileList[i])
        res = cv2.resize(pic, (300, 300), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(outpath + "1_" + str(i + 1) + ".png", res)
