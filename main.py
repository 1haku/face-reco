import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image
from PIL import ImageTk

cap = cv2.VideoCapture(0)

# 检测人脸
def detect_face(img):
    # 将测试图像转换为灰度图像，因为opencv人脸检测器需要灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 加载OpenCV人脸检测分类器Haar
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # 检测多尺度图像，返回值是一张脸部区域信息的列表（x,y,宽,高）
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    print(faces)

    # 如果未检测到面部，则返回原始图像
    if (len(faces) == 0):
        return None, None
    # 目前假设只有一张脸，xy为左上角坐标，wh为矩形的宽高
    (x, y, w, h) = faces[0]

    # 返回图像的正面部分
    return gray[y:y + w, x:x + h], faces[0]

# 该函数将读取所有的训练图像，从每个图像检测人脸并将返回两个相同大小的列表，分别为脸部信息和标签

def prepare_training_data(data_folder_path):
    dirs = os.listdir(data_folder_path)
    faces = []
    labels = []

    for dir_name in dirs:
        # 检查目录名称是否为整数
        if not dir_name.isdigit():
            continue

        label = int(dir_name)
        subject_dir_path = data_folder_path + "/" + dir_name
        subject_images_names = os.listdir(subject_dir_path)

        for image_name in subject_images_names:
            image_path = subject_dir_path + "/" + image_name
            image = cv2.imread(image_path)

            # 在读取图像之前，确保图像不为空
            if image is not None:
                face, rect = detect_face(image)
                if face is not None:
                    faces.append(face)
                    labels.append(label)
    return faces, labels


# 创建LBPH识别器并开始训练
face_recognizer = cv2.face_LBPHFaceRecognizer.create()
faces, labels = prepare_training_data("training_data")
face_recognizer.train(faces, np.array(labels))

def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (128, 128, 0), 2)

# 根据给定的（x，y）坐标标识出人名
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (128, 128, 0), 2)

# 建立标签与人名的映射列表
subjects = ["jaychou", "messi","jj lin","cho"]

# 此函数识别传递的图像中的人物并在检测到的脸部周围绘制一个矩形及其名称
def predict(test_img):
    # 生成图像的副本，这样就能保留原始图像
    img = test_img.copy()
    # 检测人脸
    face, rect = detect_face(img)
    # 预测人脸
    label = face_recognizer.predict(face)
    # 获取由人脸识别器返回的相应标签的名称
    label_text = subjects[label[0]]

    # 在检测到的脸部周围画一个矩形
    draw_rectangle(img, rect)
    # 标出预测的名字
    draw_text(img, label_text, rect[0], rect[1] - 5)
    # 返回预测的图像
    print(f"Predicted label: {label[0]}")
    return img

# 创建主窗口
window = tk.Tk()
window.title("人脸识别GUI")

# 添加标签
label = tk.Label(window, text="选择测试图像:")
label.pack()

# 添加按钮，用于加载测试图像和执行识别
def load_and_recognize_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        test_img = cv2.imread(file_path)
        predicted_img = predict(test_img)

        # 将OpenCV图像转换为Tkinter PhotoImage
        predicted_img = cv2.cvtColor(predicted_img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(predicted_img)
        img_tk = ImageTk.PhotoImage(image=img)

        # 显示预测结果
        result_label.config(image=img_tk)
        result_label.image = img_tk

load_button = tk.Button(window, text="选择测试图像", command=load_and_recognize_image)
load_button.pack()

# 添加标签，用于显示识别结果
result_label = tk.Label(window)
result_label.pack()

def show_subjects():
    subjects_label.config(text="当前标签与人名的映射：\n" + "\n".join([f"{i}: {subject}" for i, subject in enumerate(subjects)]))

# 创建一个标签，用于显示当前标签与人名的映射
subjects_label = tk.Label(window, text="当前标签与人名的映射：\n" + "\n".join([f"{i}: {subject}" for i, subject in enumerate(subjects)]))
subjects_label.pack()


# 添加输入框，用于修改subjects列表
input_label = tk.Label(window, text="修改人名标签:")
input_label.pack()

input_entry = tk.Entry(window)
input_entry.pack()
def update_subjects():
    new_label = input_entry.get()
    if new_label:
        subjects.append(new_label)
        input_entry.delete(0, "end")  # 清空输入框

update_button = tk.Button(window, text="添加新标签", command=update_subjects)
update_button.pack()

# 创建一个按钮，用于显示当前标签与人名的映射
show_subjects_button = tk.Button(window, text="显示标签与人名", command=show_subjects)
show_subjects_button.pack()


def real_time_face_detection():
    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 转换为灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 检测多个人脸
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # 提取每个人脸区域
            face = gray[y:y + h, x:x + w]

            # 进行人脸识别
            label = face_recognizer.predict(face)

            # 获取人脸识别的标签
            label_text = subjects[label[0]]

            # 在图像上绘制人名
            draw_text(frame, label_text, x, y - 10)

            # 在图像上绘制矩形框
            draw_rectangle(frame, (x, y, w, h))

        # 显示实时图像
        cv2.imshow("Real-time Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 在你的GUI中添加一个按钮，点击后触发实时检测
real_time_detection_button = tk.Button(window, text="启动实时人脸检测", command=real_time_face_detection)
real_time_detection_button.pack()

# 保持主循环运行
window.mainloop()

# 释放摄像头
cap.release()
cv2.destroyAllWindows()
