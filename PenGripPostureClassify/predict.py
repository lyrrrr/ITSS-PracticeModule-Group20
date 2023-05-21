import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
import pickle

# 加载模型和类别映射
model = load_model('my_model.h5')
with open('class_indices.pkl', 'rb') as f:
    class_indices = pickle.load(f)
class_names = list(class_indices.keys())

# 使用摄像头作为视频输入
video = cv2.VideoCapture(0)

# 剩余的代码...
# 定义预处理函数
def preprocess_frame(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = img_to_array(frame)
    frame = np.expand_dims(frame, axis=0)
    frame = preprocess_input(frame)
    return frame

fps = video.get(cv2.CAP_PROP_FPS)
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_file = 'new.mp4'  # replace with your desired output video file
output_video = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# 定义框的位置和大小
box_x = 50
box_y = 50
box_width = 200
box_height = 100

# 主循环
while True:
    # 读取一帧
    ret, frame = video.read()

    # 如果读取失败（例如，因为视频结束了），则退出循环
    if not ret:
        break

    # 预处理帧并进行预测
    preprocessed_frame = preprocess_frame(frame)
    prediction = model.predict(preprocessed_frame)
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = class_names[predicted_class_index]
    if predicted_class_name == "A":  # replace with the correct class name
        color = (0, 255, 0)  # green
    else:
        color = (0, 0, 255)  # red

    # 在帧上显示预测结果
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_width, box_y + box_height), color, thickness=2)
    cv2.putText(frame, predicted_class_name, (box_x, box_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # 实时显示预测结果和视频帧
    cv2.imshow('Video', frame)

    # 如果用户按下了 'q' 键，退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放视频和摄像头，关闭窗口
video.release()
cv2.destroyAllWindows()
