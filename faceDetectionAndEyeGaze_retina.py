import cv2
import numpy as np

# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
faceCascade = cv2.CascadeClassifier(
    'Cascades/haarcascade_frontalface_default.xml')
# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eyeCascade = cv2.CascadeClassifier('Cascades/haarcascade_eye.xml')

# source de la vidéo
#cap = cv2.VideoCapture("test_video.MOV")
cap = cv2.VideoCapture(0)

firstEyeGazes, secondEyeGazes = False, False


while 1:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # je considère que la personne regarde la caméra lorsque un des 2 yeux la regarde
    personGazes = firstEyeGazes or secondEyeGazes

    # convertit l'image en 50 nuances de gris
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detecte les têtes
    faces = faceCascade.detectMultiScale(grayFrame, 1.3, 5)
    # trier les têtes
    if len(faces) != 0:

        (x, y, w, h) = faces[0]

        if personGazes:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)

        firstEyeGazes, secondEyeGazes = False, False

        grayRoi = grayFrame[y:y+h, x:x+w]
        colorRoi = frame[y:y+h, x:x+w]

        eyes = eyeCascade.detectMultiScale(grayRoi)

        eyesSorted = sorted(eyes, key=lambda x: x[3]*x[2], reverse=True)

        eyeCounter = 0

        for (ex, ey, ew, eh) in eyesSorted:

            cv2.rectangle(colorRoi, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)

            # définit la région d'intérêt pour un oeil
            eyeGrayRoi = grayRoi[ey: ey+eh, ex:ex+ew]
            eyeColorRoi = colorRoi[ey:ey+eh, ex:ex+ew]

            # calcule la taille en pixels de ces cadrages
            rows, cols = eyeGrayRoi.shape

            # floute l'image obtenue afin de réduire le bruit
            blur = 7  # degré du floutage (mettre nombre impair)
            blurred_blurred = cv2.GaussianBlur(eyeGrayRoi, (blur, blur), 7)

            # sépare la séquence en noir et blanc suivant une certaine limite de nuance de gris
            _, threshold = cv2.threshold(
                eyeGrayRoi, 35, 255, cv2.THRESH_BINARY_INV)

            # renvoie des listes des coordonnées des outline des zones blanches
            outline, _ = cv2.findContours(
                threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # trie ses listes afin d'avoir le plus grand contour en premier élément
            outlineSorted = sorted(
                outline, key=lambda x: cv2.contourArea(x), reverse=True)

            # si un contour a été détecté, on renvoie la liste du plus grand contour
            if len(outlineSorted) != 0:

                outlineBis = outlineSorted[0]

                (cx, cy, cw, ch) = cv2.boundingRect(outlineBis)
                cv2.rectangle(eyeColorRoi, (cx, cy),
                              (cx+cw, cy+ch), (255, 0, 0), 2)

                abscisse_du_centre = cx+int(cw/2)
                ordonnee_du_centre = cy+int(ch/2)
                pas_x = int(ew/50)
                pas_y = int(eh/50)

                # problème de coordonnées
                if eyeCounter == 0:

                    firstEyeGazes = (24*pas_x < abscisse_du_centre < ew-24 *
                                     pas_x) and (24*pas_y < ordonnee_du_centre < eh-24*pas_y)

                else:
                    secondEyeGazes = (24*pas_x < abscisse_du_centre < ew-24 *
                                      pas_x) and (24*pas_y < ordonnee_du_centre < eh-24*pas_y)

            eyeCounter += 1
            if eyeCounter >= 2:
                break

        cv2.imshow("frame", frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
