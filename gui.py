from tkinter import *
from predict import predict
import cv2
from tkinter.simpledialog import askstring
from tkinter import messagebox
import os
import uuid
import time
from training import train
root = Tk()
root.title('简单的人脸识别系统')


# 窗口放置中心位置
def center_window(root, width, height):
    screenwidth = root.winfo_screenwidth()
    screenheight = root.winfo_screenheight()
    size = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
    print(size)
    root.geometry(size)


# 按帧采集摄图像进行预测
def open_button_event():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        predict(frame)
        cv2.imshow("点击esc 退出", frame)
        c = cv2.waitKey(1)
        if c == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


def cap_picture_button_event():
    tarin_data_path = r'data/train/'
    val_data_path = r'data/val/'
    ask_window = askstring("提示：", "请输入采集者姓名(拼音，不然训练时会出问题)")  # 获得输入的字符串
    print(ask_window)
    # 应为图片是放在文件夹里，所以要去判别采集者的文件是否存在
    tarin_data_path = tarin_data_path + ask_window
    val_data_path = val_data_path + ask_window
    # 校验文件夹是否存在
    if os.path.exists(tarin_data_path):
        print("已采集过，不需要重新采集")
        messagebox.showinfo("警告", '已采集过，不需要重新采集')

    # 校验通过
    # 创建文件夹，并把采集到得图片放入val和train文件夹里
    #
    os.mkdir(tarin_data_path)
    os.mkdir(val_data_path)

    cap = cv2.VideoCapture(0)
    start = time.time()
    index = 0
    while True:
        ret, frame = cap.read()
        cv2.imshow("f", frame)
        c = cv2.waitKey(1000)  # 每个三秒采集一次
        uid = str(uuid.uuid4())
        # cv2.imwrite(tarin_data_path + '/' + uid + '.jpg', frame)
        cv2.imencode('.jpg', frame)[1].tofile(tarin_data_path + '/' + uid + '.jpg')
        index += 1
        if (index + 1) % 3 == 0:
            cv2.imencode('.jpg', frame)[1].tofile(val_data_path + '/' + uid + '.jpg')
            # cv2.imwrite(val_data_path + '/' + uid + '.jpg', frame)
        endtime = time.time() - start
        if c == 27 or endtime >= 20:
            break
    cap.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("提示", '采集完毕！')


# 定义botton
# 开机按钮
def re_train_button():
    messagebox.showinfo("提示", '采集完毕！')
    train()

lbl = Label(root, text='功能：', font=("Arial Bold", 20)).grid(column=0, row=0)
open_button = Button(root, text='启动', font=("Arial Bold", 10), command=open_button_event).grid(column=1, row=1)
cap_picture_button = Button(root, text='采集', font=("Arial Bold", 10), command=cap_picture_button_event).grid(column=2,
                                                                                                             row=1)
# 采集完模型需要重新训练
re_train_button = Button(root, text='重新训练', font=("Arial Bold", 10),command=re_train_button).grid(column=3, row=1)
center_window(root, 300, 240)
root.maxsize(600, 400)
root.minsize(300, 240)
root.mainloop()
