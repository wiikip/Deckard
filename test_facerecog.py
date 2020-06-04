import face_recognition
import pickle
from os import listdir
from os.path import isfile, join
import numpy as np


all_face_encodings = {}

onlyfiles = [f for f in listdir("bdd_train") if isfile(join("bdd_train", f))][:]

i=0
for elm in onlyfiles:
    img = face_recognition.load_image_file("bdd_train/"+elm)
    all_face_encodings[elm.split('.')[0]] = face_recognition.face_encodings(img)[0]
    print(i, elm, face_recognition.face_encodings(img)[0] == [])
    i=i+1

with open('dataset_faces_test.dat', 'wb') as f:
    pickle.dump(all_face_encodings, f)


bdd_test = [f for f in listdir("bdd_test")]

with open('dataset_faces_test.dat', 'rb') as f:
	all_face_encodings = pickle.load(f)

known_face_names = list(all_face_encodings.keys())
known_face_encodings = np.array(list(all_face_encodings.values()))

yes, no = 0, 0

for elm in bdd_test:
    unknown_img = face_recognition.load_image_file("bdd_test/"+elm)
    if face_recognition.face_encodings(unknown_img) == []:
        continue
    else:
        unknown_encoding = face_recognition.face_encodings(unknown_img)[0]
    results = face_recognition.compare_faces(known_face_encodings, unknown_encoding)
    res = False
    for a in results:
        res = res or a
    if res == True:
        yes += 1
    else:
        no +=1
    print('Accuracy =', yes*100/(yes+no), '%')
    # 7438 18 7456