import cv2
import face_recognition
import numpy as np
import pickle
import time
from multiprocessing import Process, Queue
import pyttsx3
import speech_recognition as sr



def camera_live(names_queue, encodings_queue, new_names_queue):
    video_capture = cv2.VideoCapture(0)

    while True:
        with open('dataset_faces.dat', 'rb') as f:
            all_face_encodings = pickle.load(f)

        known_face_names = list(all_face_encodings.keys())
        known_face_encodings = np.array(list(all_face_encodings.values()))

        face_locations = []
        face_encodings = []
        face_names = []
        process_this_frame = True

        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

        process_this_frame = not process_this_frame
        names_queue.put(name)
        encodings_queue.put(face_recognition.face_encodings(frame)[0])

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()


def voice_live(names_queue, encodings_queue, new_names_queue) :
    engine = pyttsx3.init()
    text_save = ''
    r = sr.Recognizer()
    while True:
        text = names_queue.get()
        if text != text_save and text != 'Unknown':
            engine.say('Salut ' + text + 'comment allez-vous ?')
            engine.runAndWait()
            with sr.Microphone() as source:
                audio = r.listen(source)
                rep = r.recognize_google(audio, language="fr-FR")
                print(rep)
        if text != text_save and text == 'Unknown':
            engine.say('Vous êtes qui ?')
            engine.runAndWait()
            with sr.Microphone() as source:
                audio = r.listen(source)
                try:
                    rep = r.recognize_google(audio, language="fr-FR")
                    with open('dataset_faces.dat', 'rb') as f:
                        all_face_encodings = pickle.load(f)

                    all_face_encodings[rep] = encodings_queue.get()

                    with open('dataset_faces.dat', 'wb') as f:
                        pickle.dump(all_face_encodings, f)    
                except:
                    engine.say('Je n\'ai pas compris')
                    engine.runAndWait()




        text_save = text

if __name__ == '__main__':

    temps = time.time()
    names_queue = Queue()
    encodings_queue = Queue()
    new_names_queue = Queue()
    p1 = Process(target=camera_live, args=(names_queue, encodings_queue, new_names_queue))
    p2 = Process(target=voice_live, args=(names_queue, encodings_queue, new_names_queue))

    p1.start()
    p2.start()
    p1.join()
    p2.join()

