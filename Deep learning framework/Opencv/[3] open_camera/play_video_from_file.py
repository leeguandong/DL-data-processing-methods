import cv2

# 播放本地视频
capture = cv2.VideoCapture(
    'F:/Github/DL-data-processing-methods/Deep learning framework/Opencv/[3] open_camera/output.avi')

while (capture.isOpened()):
    ret, frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame', gray)
    if cv2.waitKey(30) == ord('q'):
        break
