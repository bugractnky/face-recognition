# 🧠 Yüz Tanıma Sistemi (Face Recognition Project)

Bu proje, Python kullanılarak geliştirilmiş bir yüz tanıma sistemidir. `dlib`, `face_recognition`, `OpenCV` gibi kütüphanelerle gerçek zamanlı yüz tanıma, kayıt ve doğrulama işlemleri yapılmaktadır. Ayrıca Telegram bot entegrasyonu da mevcuttur.

## 🚀 Dosyalar
- `train_faces.py`: Klasör içerisindeki yüzleri eğitir, veri seti oluşturur.
- `train_faces_dlib.py`: Alternatif dlib tabanlı eğitim scripti.
- `face_unlock.py`: Kamera üzerinden yüz tanıma ile "kilit açma" fonksiyonu.
- `telegram_face_bot.py`: Telegram üzerinden yüz tanıma işlevi sunar.
- `face_data/`: Tanıtılmış yüz fotoğraflarının bulunduğu klasör.

## 🧰 Gereksinimler
```bash
pip install face_recognition opencv-python dlib telepot


# Yüz eğitimi
python3 train_faces.py

# Kamera ile tanıma
python3 face_unlock.py

# Telegram bot (token gerekli)
python3 telegram_face_bot.py
