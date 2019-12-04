import cv2
import numpy as np

cap = cv2.VideoCapture("test_video.MOV")

while 1:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 0)

    roi = frame[360: 490, 1000: 1170]
    rows, cols, _ = roi.shape
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    rows_droite, cols_droite, _ = roi_droite.shape
    rows_gauche, cols_gauche, _ = roi_gauche.shape
    

    contours, _ = cv2.findContours( threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        #cv2.drawContours(roi, [cnt], -1, (0, 0, 255), 3)
        #cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.line(roi, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 2)
        cv2.line(roi, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 2)
    #cv2.imshow('threshold',threshold)
    #cv2.imshow('gray_roi',gray_roi)
    cv2.imshow('roi',roi)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
       break
            
cap.release()
cv2.destroyAllWindows()
