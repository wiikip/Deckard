import speech_recognition as sr
import pyttsx3

r = sr.Recognizer()
engine = pyttsx3.init()
'''
engine.say('Salut çava la famille et tout tu abuses frère wallah')
engine.runAndWait()
'''
with sr.Microphone() as source:
    print('Gol chi 7aja :')
    audio = r.listen(source)
    try:
        text = r.recognize_google(audio, language="fr-FR")
        print(text)
    except:
        print('mafhmatch')
