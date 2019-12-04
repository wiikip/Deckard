import cv2
import numpy as np

cap = cv2.VideoCapture("test_video.MOV")

while 1:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 0)

    roi_droite = frame[360: 490, 1000: 1170]
    roi_gauche = frame[360:490, 800:960]

    rows_droite, cols_droite, _ = roi_droite.shape
    rows_gauche, cols_gauche, _ = roi_gauche.shape
    

    gray_roi_droite = cv2.cvtColor(roi_droite, cv2.COLOR_BGR2GRAY)
    gray_roi_gauche = cv2.cvtColor(roi_gauche, cv2.COLOR_BGR2GRAY)

    gray_roi_droite_blurred = cv2.GaussianBlur(gray_roi_droite, (7, 7), 7)
    gray_roi_gauche_blurred = cv2.GaussianBlur(gray_roi_gauche, (7, 7), 7)

    _, threshold_droite = cv2.threshold(gray_roi_droite_blurred, 3, 255, cv2.THRESH_BINARY_INV)
    _, threshold_gauche = cv2.threshold(gray_roi_gauche_blurred, 3, 255, cv2.THRESH_BINARY_INV)

    contours_droite, _ = cv2.findContours( threshold_droite, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_gauche, _ = cv2.findContours( threshold_gauche, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    contours_droite_sorted = sorted(contours_droite, key=lambda x: cv2.contourArea(x), reverse=True)
    contours_gauche_sorted = sorted(contours_gauche, key=lambda x: cv2.contourArea(x), reverse=True)


    for cnt_d in contours_droite_sorted:
        (x_d, y_d, w_d, h_d) = cv2.boundingRect(cnt_d)

        #cv2.drawContours(roi, [cnt], -1, (0, 0, 255), 3)
        #cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.line(roi_droite, (x_d + int(w_d/2), 0), (x_d + int(w_d/2), rows_droite), (0, 255, 0), 2)
        cv2.line(roi_droite, (0, y_d + int(h_d/2)), (cols_droite, y_d + int(h_d/2)), (0, 255, 0), 2)
   
    for cnt_g in contours_gauche_sorted:
        (x_g, y_g, w_g, h_g) = cv2.boundingRect(cnt_g)
        
        #cv2.drawContours(roi, [cnt], -1, (0, 0, 255), 3)
        #cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        cv2.line(roi_gauche, (x_g + int(w_g/2), 0), (x_g + int(w_g/2), rows_gauche), (0, 255, 0), 2)
        cv2.line(roi_gauche, (0, y_g + int(h_g/2)), (cols_gauche, y_g + int(h_g/2)), (0, 255, 0), 2)


    cv2.imshow('roi_droite',roi_droite)
    cv2.imshow('roi_gauche',roi_gauche)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
       break
            
cap.release()
cv2.destroyAllWindows()
