import os
import cv2
from keras.models import model_from_json
import numpy as np
from mtcnn.mtcnn import MTCNN  # 用于检测和裁剪人脸

# 加载模型结构
with open('facenet_keras.json', 'r') as json_file:
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)

# 加载权重
loaded_model.load_weights('facenet_keras_weights.h5')

# 指定图像文件夹和关键字
image_folder = 'downloaded_images/messi'
keyword = 'human'

# 创建结果文件夹
result_folder = 'path/result_folder'
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

# 初始化MTCNN检测器
detector = MTCNN()

# 加载并筛选图像
for filename in os.listdir(image_folder):
    if keyword in filename.lower() and filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path)
        # 检测人脸
        faces = detector.detect_faces(img)
        if faces:
            for i, face in enumerate(faces):
                x, y, width, height = face['box']
                face_img = img[y:y + height, x:x + width]
                # 将人脸图像调整为模型输入大小
                face_img = cv2.resize(face_img, (160, 160))
                face_img = np.expand_dims(face_img, axis=0)
                # 使用FaceNet模型进行特征提取
                features = model.predict(face_img)
                # 在结果文件夹中保存特征向量
                feature_file = os.path.join(result_folder, f'feature_{i}_{filename}.npy')
                np.save(feature_file, features)

print("筛选完成")