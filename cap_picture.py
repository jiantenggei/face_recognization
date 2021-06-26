import os
import cv2
import time
import uuid
cap = cv2.VideoCapture(0)
start = time.time()
person_name = input("请输入人名(拼音)：")
data_path = 'data/train/' + person_name
if os.path.exists(data_path):
    print("已采集过，不需要重新采集")
    exit(0)
os.mkdir(data_path)
while True:
    ret, frame = cap.read()
    cv2.imshow("f", frame)
    c = cv2.waitKey(1000)  # 每个三秒采集一次
    uid=str(uuid.uuid4())
    cv2.imwrite(data_path+'/'+uid+'.jpg',frame)
    endtime=time.time()-start
    if c == 27 or endtime>=20:
        break
