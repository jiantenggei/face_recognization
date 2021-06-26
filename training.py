from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder  # 将离散的数据转换到0~classes-1
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from get_feature import get_feature
import pickle

def train():
    X_train, y_train, X_test, y_test = get_feature()
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    # 样本的特征值除以个特征值的平方和，归一化
    in_encoder = Normalizer()
    X_train = in_encoder.transform(X_train)
    X_test = in_encoder.transform(X_test)
    # lable_encoder
    out_encoder = LabelEncoder()
    out_encoder.fit(y_train)
    y_train = out_encoder.transform(y_train)
    y_test = out_encoder.transform(y_test)

    model = SVC(kernel='linear', probability=True)

    model.fit(X_train, y_train)

    # 预测，predict
    yhat_train = model.predict(X_train)
    yhat_test = model.predict(X_test)

    score_train = accuracy_score(y_train, yhat_train)
    score_test = accuracy_score(y_test,yhat_test)

    print('Accuracy: train=%.3f, test=%.3f' % (score_train * 100, score_test * 100))

    #模型的保存
    pkl_filename = "sklearn_model.pkl"
    with open(pkl_filename,'wb') as f:
        pickle.dump(model,f)

if __name__ == '__main__':
    train()
