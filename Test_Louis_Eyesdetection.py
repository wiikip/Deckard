import numpy as np
import cv2

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('Cascades/haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    img = cv2.flip(img,1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            for i in range(ew/2+1):
                for j in range(eh/2+1):
                    
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()


'''
gray_eye = eye[min_y: max_y, min_x: max_x]
_, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
height, width = threshold_eye.shape
left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
left_side_white = cv2.countNonZero(left_side_threshold)
right_side_threshold = threshold_eye[0: height, int(width / 2): width]
right_side_white = cv2.countNonZero(right_side_threshold)


gaze_ratio = left_side_white / right_side_white


# Gaze detection
gaze_ratio_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
gaze_ratio_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)
gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2


if gaze_ratio <= 1:
    cv2.putText(frame, "RIGHT", (50, 100), font, 2, (0, 0, 255), 3)
    new_frame[:] = (0, 0, 255)
elif 1 < gaze_ratio < 1.7:
    cv2.putText(frame, "CENTER", (50, 100), font, 2, (0, 0, 255), 3)
else:
    new_frame[:] = (255, 0, 0)
    cv2.putText(frame, "LEFT", (50, 100), font, 2, (0, 0, 255), 3)




'''




