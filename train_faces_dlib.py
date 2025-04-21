import dlib
import cv2
import numpy as np
import os
import pickle

# Modellerin bulunduğu yer
shape_predictor_path = "/usr/share/dlib/shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "/usr/share/dlib/dlib_face_recognition_resnet_model_v1.dat"

# Klasör ve isim listeleri
dataset_dir = "/home/bugra/face_data"
encodings = []
names = []

# Modelleri yükle
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(shape_predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

# Yüzleri oku
for filename in os.listdir(dataset_dir):
    if filename.endswith(".jpg") or filename.endswith(".jpeg"):
        path = os.path.join(dataset_dir, filename)
        name = os.path.splitext(filename)[0]

        img = cv2.imread(path)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dets = detector(rgb, 1)

        for d in dets:
            shape = sp(rgb, d)
            face_descriptor = facerec.compute_face_descriptor(rgb, shape)
            encodings.append(np.array(face_descriptor))
            names.append(name)
            print(f"✅ Yüz alındı: {name}")

# Encode edilen yüzleri kaydet
data = {"encodings": encodings, "names": names}
with open("trained_faces_dlib.pkl", "wb") as f:
    pickle.dump(data, f)

print("🎯 Eğitim tamamlandı. Model: trained_faces_dlib.pkl")
