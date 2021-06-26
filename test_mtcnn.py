import tensorflow as tf
from mtcnn import MTCNN
import cv2

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
test_img = cv2.imread("imgs/img.png")
detector = MTCNN()
result=detector.detect_faces(test_img)
print(result[0]['box'])
for item in result:
    x,y,width,height = item['box']
    confidence=item['confidence']
    confidence=round(confidence,3)
    test_img = cv2.rectangle(test_img,(x,y),(x+width,y+height),color=(0,255,0),thickness=2)
    test_img = cv2.putText(test_img,str(confidence),(x,y-10),color=(0,0,255),fontScale=3,fontFace=cv2.FONT_HERSHEY_PLAIN,thickness=2)

cv2.imshow('img',test_img)
cv2.waitKey()