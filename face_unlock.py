import face_recognition
print("🧠 face_recognition import edildi")
import face_recognition_models
import os

# MODELLERİ ZORLA GÖSTER
os.environ["FACE_RECOGNITION_MODEL_LOCATION"] = "/home/bugra/jarvis_env/lib/python3.12/site-packages/face_recognition_models/models"
face_recognition_models.model_location = lambda: os.environ["FACE_RECOGNITION_MODEL_LOCATION"]

import cv2
import pyttsx3
import time


# Sesli konuşma
engine = pyttsx3.init()
engine.setProperty("rate", 170)
engine.setProperty("voice", "english")

def speak(text):
    print(f"🤖 Jarvis: {text}")
    engine.say(text)
    engine.runAndWait()

# Kayıtlı yüz (Buğra)
known_image = face_recognition.load_image_file("/home/bugra/jarvis_face/bugra.jpeg")
known_encoding = face_recognition.face_encodings(known_image)[0]

# Kamera başlat
video = cv2.VideoCapture(0)

speak("Face verification active. Please look at the camera.")

found = False
start_time = time.time()

while True:
    ret, frame = video.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        match = face_recognition.compare_faces([known_encoding], face_encoding)
        if match[0]:
            speak("Identity verified. Welcome Buğra.")
            found = True
            break

    if found or time.time() - start_time > 15:
        break

video.release()
cv2.destroyAllWindows()

if not found:
    speak("Access denied. I could not verify your identity.")
