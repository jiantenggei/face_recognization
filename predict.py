from sklearn.preprocessing import Normalizer
from extract_face.extract_face import extract_faces
import cv2
import pickle
from tensorflow.keras.models import load_model
import numpy as np
from get_feature import get_embedding, create_dict

# 预测跟训练类
# 1.打开摄像头，mtcnn获取人脸
# 2.facenet 计算特征向量
# 放入svm中进行分类 最后输出预测结果
dict = create_dict(r'data/train/')


def predict(image):
    global dict
    print(dict)
    face_list, boxes_list = extract_faces(image)
    if len(face_list) != 0:
        facenet_model = load_model('weights/facenet_keras.h5')
        emdTrainX = list()
        for face in face_list:
            emd = get_embedding(facenet_model, face)
            emdTrainX.append(emd)

        emdTrainX = np.asarray(emdTrainX)
        emdTrainX.reshape(-1, 1)
        in_encoder = Normalizer()
        X_train = in_encoder.transform(emdTrainX)
        pkl_filename = "sklearn_model.pkl"
        with open(pkl_filename, 'rb') as file:
            pickle_model = pickle.load(file)
        result_list = pickle_model.predict_proba(X_train)
        print(result_list)
        print(type(result_list))
        result_index = np.argmax(result_list, axis=1)
        print(result_list)
        index = 0
        for result in result_index:
            if result_list[index][result] <= 0.6:
                label = 'unkonw'
            else:
                label = dict[result]
            x, y, width, height = boxes_list[index]
            image = cv2.rectangle(image, (x, y), (x + width, y + height), color=(0, 255, 0), thickness=2)
            image = cv2.putText(image, str(label), (x, y - 10), color=(0, 0, 255), fontScale=1,
                                fontFace=cv2.FONT_HERSHEY_PLAIN, thickness=2)
            index += 1
    return image


if __name__ == '__main__':
    # cap = cv2.VideoCapture(0)
    # while True:
    #     ret, frame = cap.read()
    #     predict(frame)
    #     cv2.imshow("f", frame)
    #     c=cv2.waitKey(30)
    #     if c == 27 :
    #         break
    # cap.release()
    # cv2.destroyAllWindows()
    img = cv2.imread("imgs/img_3.png")
    predict(img)
    cv2.imshow('t', img)
    cv2.waitKey()
