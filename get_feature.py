import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
from load_data.load_data import load_dataset
import os
def get_embedding(model, face):
    face = face.astype('float32')
    # 归一化
    mean, std = face.mean(), face.std()
    face = (face - mean) / std
    print(face.shape)
    sample = np.expand_dims(face, axis=0)
    yhat = model.predict(sample)
    print(yhat[0])

    return yhat[0]


def get_feature():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    X_train, y_train = load_dataset('data/train/')
    # print(X_train.shape)
    # print(y_train.shape)
    X_val, y_val = load_dataset('data/val/')
    # print(X_val.shape)
    # print(y_val.shape)
    np.savez_compressed('data/data-face-dataset.npz', X_train, y_train, X_val, y_val)
    data = np.load('data/data-face-dataset.npz')
    trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)
    facenet_model = load_model('weights/facenet_keras.h5')
    emdTrainX = list()
    for face in trainX:
        emd = get_embedding(facenet_model, face)
        emdTrainX.append(emd)

    emdTrainX = np.asarray(emdTrainX)

    emdTestX = list()
    for face in testX:
        emd = get_embedding(facenet_model, face)
        emdTestX.append(emd)
    emdTestX = np.asarray(emdTestX)
    #np.savez_compressed('faces-embeddings.npz', emdTrainX, trainy, emdTestX, testy)
    print(emdTestX.shape, emdTrainX.shape)
    return emdTrainX,trainy, emdTestX,testy

def create_dict(path):
    #os.listdir拿到相应的文件  拿到相应文件夹的名字 X
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    itype={}
    for index, folder in enumerate(cate):
            itype[index] =folder.split('/')[-1]


    return itype


if __name__ == '__main__':
    dict=create_dict('data/train/')
    print(dict)