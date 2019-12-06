import cv2
import numpy as np

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('Cascades/haarcascade_eye.xml')

#source de la vidéo
cap = cv2.VideoCapture("test_video.mov")

while 1:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 0)


    #convertit l'image en 50 nuances de gris
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    
    #detecte les têtes
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

    if len(faces)!=0:
        (x,y,w,h) = faces[0]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
        
        gray_roi = gray_frame[y:y+h, x:x+w]
        color_roi = frame[y:y+h,x:x+w]
        
        eyes = eye_cascade.detectMultiScale(gray_roi)
        eyes_sorted=sorted(eyes, key= lambda x:x[3]*x[2],reverse=True)

        #le compteur sert à ne prendre en compte que les deux premiers objets détectés comme des yeux 
        i=0
        for (ex,ey,ew,eh) in eyes_sorted:
            #cv2.rectangle(color_roi,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            
            #définit la région d'intérêt pour un oeil
            eye_gray_roi = gray_roi[ey: ey+eh, ex:ex+ew]
            eye_color_roi = color_roi[ey:ey+eh,ex:ex+ew]

            #calcule la taille en pixels de ces cadrages
            rows, cols = eye_gray_roi.shape
        
            #floute l'image obtenue afin de réduire le bruit
            blur=7 #degré du floutage (mettre nombre impair)
            blurred_blurred = cv2.GaussianBlur(eye_gray_roi, (blur, blur), 7)


            #sépare la séquence en noir et blanc suivant une certaine limite de nuance de gris
            _, threshold = cv2.threshold(eye_gray_roi, 3, 255, cv2.THRESH_BINARY_INV)

            #renvoie des listes des coordonnées des contours des zones blanches
            contours, _ = cv2.findContours( threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            #trie ses listes afin d'avoir le plus grand contour en premier élément
            contours_sorted = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

            #si un contour a été détecté, on renvoie la liste du plus grand contour
            if not len(contours_sorted)==0:

                cnt=contours_sorted[0]

                (x, y, w, h) = cv2.boundingRect(cnt)

                cv2.line(eye_color_roi, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 2)
                cv2.line(eye_color_roi, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 2)
            
            i+=1
            if i>=2:
                break
    
            
                
        cv2.imshow("frame",frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
       break
            
cap.release()
cv2.destroyAllWindows()
