import face_recognition
import os
import pickle

# Yüz klasörü
dataset_dir = "/home/bugra/face_data"
known_encodings = []
known_names = []

for filename in os.listdir(dataset_dir):
    if filename.endswith(".jpg") or filename.endswith(".jpeg"):
        path = os.path.join(dataset_dir, filename)
        name = os.path.splitext(filename)[0]

        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)

        if len(encodings) > 0:
            known_encodings.append(encodings[0])
            known_names.append(name)
            print(f"✅ Yüz verisi alındı: {name}")
        else:
            print(f"❌ Yüz bulunamadı: {filename}")

# Modeli kaydet
data = {"encodings": known_encodings, "names": known_names}
with open("trained_faces.pkl", "wb") as f:
    pickle.dump(data, f)

print("🎯 Eğitim tamamlandı. Model kaydedildi: trained_faces.pkl")
