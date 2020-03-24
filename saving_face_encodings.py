import face_recognition
import pickle

all_face_encodings = {}

from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir("Data") if isfile(join("Data", f))]

for elm in onlyfiles:
    img = face_recognition.load_image_file("Data\\"+elm)
    all_face_encodings[elm.split('.')[0]] = face_recognition.face_encodings(img)[0]

with open('dataset_faces.dat', 'wb') as f:
    pickle.dump(all_face_encodings, f)