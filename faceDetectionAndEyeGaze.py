import cv2
import numpy as np

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('Cascades/haarcascade_eye.xml')

#source de la vidéo
cap = cv2.VideoCapture("test_video.MOV")

oeil1_me_regarde,oeil2_me_regarde=False,False


while 1:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 0)

    #je considère que la personne regarde la caméra lorsque un des 2 yeux la regarde
    la_personne_me_regarde = oeil1_me_regarde or oeil2_me_regarde
    
    #convertit l'image en 50 nuances de gris
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detecte les têtes
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    #trier les têtes
    if len(faces)!=0:

        (x,y,w,h) = faces[0]
        
        
        if la_personne_me_regarde:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)

        else:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
        
        
        #oeil1_me_regarde,oeil2_me_regarde=False,False

        gray_roi = gray_frame[y:y+h, x:x+w]
        color_roi = frame[y:y+h,x:x+w]
        
        eyes = eye_cascade.detectMultiScale(gray_roi)
        eyes_sorted=sorted(eyes, key= lambda x:x[3]*x[2],reverse=True)

        compteur_yeux=0
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
            if len(contours_sorted)!=0:
                
                cnt=contours_sorted[0]
                
                (x, y, w, h) = cv2.boundingRect(cnt)
                cv2.rectangle(eye_color_roi,(x,y),(x+w,y+h),(0,255,0),2)



                abscisse_du_centre=x+int(w/2)
                ordonnee_du_centre=y+int(h/2)
                #problème de coordonnées
                if compteur_yeux==0:
                    
                    oeil1_me_regarde=(ex<abscisse_du_centre<ex+ew) and (ey<ordonnee_du_centre<ey+eh)
                    #print(oeil1_me_regarde)
                    
                else:
                    oeil2_me_regarde=(ex<abscisse_du_centre<ex+ew) and (ey<ordonnee_du_centre<ey+eh)


            compteur_yeux+=1
            if compteur_yeux>=2:
                break
            
                
        cv2.imshow("frame",frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
       break
            
cap.release()
cv2.destroyAllWindows()
