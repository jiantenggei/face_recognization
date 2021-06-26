import os

import cv2
import numpy as np
from extract_face.extract_face import extract_face

import warnings

warnings.filterwarnings("ignore")


# 提取制定路劲下的图片的所有人脸
def load_face(dir):
    faces = list()
    for filename in os.listdir(dir):
        path = dir + filename
        face = extract_face(path)
        if face is not None:
            faces.append(face)

    return faces


# 这里是将数据记载到内存中
# 全部加载进来，对计算机内存有一定要求
def load_dataset(dir):
    data = []
    labels = []
    for subdir in os.listdir(dir):
        path = dir + subdir + '/'
        faces = load_face(path)
        label = [subdir for i in range(len(faces))]
        print('load {} samples for People {}'.format(len(faces), subdir))
        data.extend(faces)
        labels.extend(label)
    return np.asarray(data), np.asarray(labels)


if __name__ == '__main__':
    # -------------------------------------------#
    #      打包数据集                              #
    #                                            #
    # -------------------------------------------#
    #X_train, y_train = load_dataset('../data/train/')
    # print(X_train.shape)
    # print(y_train.shape)
    # X_val, y_val = load_dataset('../data/val/')
    # print(X_val.shape)
    # print(y_val.shape)
    # np.savez_compressed('data-face-dataset.npz', X_train, y_train, X_val, y_val)

    # -------------------------------------------#
    #      测试打包好的数据集                              #
    #                                            #
    # -------------------------------------------#
    data = np.load('data-face-dataset.npz')
    trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)
