from PIL import Image
import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings

warnings.filterwarnings("ignore")


# 截取单张人脸 用于训练
def extract_face(filename, required_size=(160, 160)):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    # load image from file
    image = cv2.imread(filename)
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(image)
    # extract the bounding box from the first face
    if len(results) == 0:
        with open('test.txt', 'a') as f: #记录识别不了的人脸
            f.writelines(filename + '\n')
        return None
    try:
        x1, y1, width, height = results[0]['box']
        # deal with negative pixel index
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = image[y1:y2, x1:x2]
        # resize pixels to the model size
        face = cv2.resize(face, required_size)
        return face
    except:
        print(results)


# 截取多张人脸，用于测试摄像头
def extract_faces(image, required_size=(160, 160)):
    faces_list = []  # 存放截取后的人脸
    boxes_list = []
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    detector = MTCNN()
    result = detector.detect_faces(image)
    for item in result:
        x, y, width, height = item['box']
        boxes_list.append(item['box'])
        x1, y1 = abs(x), abs(y)
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = image[y1:y2, x1:x2]
        face = cv2.resize(face, required_size)
        faces_list.append(face)
    return faces_list, boxes_list
