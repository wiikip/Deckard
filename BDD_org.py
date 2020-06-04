from os import listdir
from os.path import isfile, join
import shutil

#Ce code permet juste de réorgnaiser la base de données en deux catégories : données pour entrainer le modèle et des donnés pour réaliser des testes


noms = [f for f in listdir("bdd")]
noms_test = []
for elm in noms:
    cpt = len([f for f in listdir("bdd/"+elm)])
    if cpt > 1 :
        noms_test += [elm]



for elm in noms_test:
    img_list = [f for f in listdir("bdd/"+elm)]
    shutil.copy('bdd/'+elm+'/'+img_list[0], 'bdd_train/')
    for i in range(1, len(img_list)):
        shutil.copy('bdd/'+elm+'/'+img_list[i], 'bdd_test/')
