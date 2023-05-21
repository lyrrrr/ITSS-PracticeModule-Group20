import cv2

# 打开默认摄像头
cap = cv2.VideoCapture(0)

while True:
    # 读取摄像头的每一帧
    ret, frame = cap.read()

    # 显示图像
    cv2.imshow("Camera", frame)

    # 按下 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()