import cv2
import os

# path = 'C:/Users/50493/Desktop/test1/'
path = './feature-like/'
outpath = './' + 'feature' + '/'

if __name__ == '__main__':
    #改文件名
    fileList = os.listdir(path)
    for i in range(1, 501):
        used_name = path + "like_" + str(i) + ".txt"
        new_name = path + "11_" + str(i) + ".txt"
        os.rename(used_name, new_name)
        # print("文件%s重命名成功,新的文件名为%s" % (used_name, new_name))
