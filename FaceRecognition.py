import cv2
import os
import numpy as np
import tkinter as tk
from PIL import Image
from PIL import ImageTk

# 创建Tkinter窗口
window = tk.Tk()
window.title("Face Recognition")

# 创建摄像头捕获对象
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # 设置宽度为800像素
cap.set(4, 480)  # 设置高度为1080像素
# 创建标签输入框
label_entry = tk.Entry(window, text="Enter label")
label_entry.pack()

# 创建录入按钮
def capture_image():
    label = label_entry.get()
    ret, frame = cap.read()
    if ret:
        # 保存图像到指定目录
        filename = f"training_data/{label}/image_{len(os.listdir(f'training_data/{label}'))}.jpg"
        cv2.imwrite(filename, frame)
        # 提示用户录入成功
        print(f"Image captured and saved as {filename}")

capture_button = tk.Button(window, text="Capture", command=capture_image)
capture_button.pack()

# 创建退出按钮
def exit_program():
    cap.release()
    window.destroy()

exit_button = tk.Button(window, text="Exit", command=exit_program)
exit_button.pack()

# 创建图像显示标签
result_label = tk.Label(window, width=640, height=480)  # 设置标签大小为所需的分辨率大小
result_label.pack()

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def update_frame():
    ret, frame = cap.read()
    if ret:
        # 将OpenCV图像转换为灰度图像以进行人脸检测
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 将OpenCV图像转换为PIL图像
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        photo = ImageTk.PhotoImage(image=image)

        # 更新标签以显示新的帧
        result_label.config(image=photo)
        result_label.image = photo

        # 递归调用以实现连续帧更新
        result_label.after(10, update_frame)

# 启动帧更新
update_frame()

# 启动Tkinter主循环
window.mainloop()