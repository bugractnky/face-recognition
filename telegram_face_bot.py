import os
import cv2
import pickle
import numpy as np
import face_recognition
from telegram.ext import Application, MessageHandler, filters
from telegram import Update
from telegram.ext import ContextTypes

# BOT TOKEN
TOKEN = "7734236319:AAHXhNjc_KXxpJvmVZgFw91q7bq-gQixl8I"

# Klasörler ve model dosyası
GUEST_DIR = "/home/bugra/jarvis_guests"
MODEL_PATH = "/home/bugra/trained_faces_dlib.pkl"

os.makedirs(GUEST_DIR, exist_ok=True)

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.message.from_user
    file = await context.bot.get_file(update.message.photo[-1].file_id)
    image_path = os.path.join(GUEST_DIR, f"{user.id}.jpg")
    await file.download_to_drive(image_path)

    # Yüz tanıma ve encode
    image = face_recognition.load_image_file(image_path)
    faces = face_recognition.face_locations(image)

    if not faces:
        await update.message.reply_text("😔 Yüz algılanamadı.")
        return

    encodings = face_recognition.face_encodings(image, known_face_locations=faces)

    if not encodings:
        await update.message.reply_text("❌ Yüz encode edilemedi.")
        return

    new_encoding = encodings[0]

    # Eski model varsa yükle
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            data = pickle.load(f)
        known_encodings = data["encodings"]
        known_names = data["names"]
    else:
        known_encodings = []
        known_names = []

    known_encodings.append(new_encoding)
    known_names.append("misafir")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"encodings": known_encodings, "names": known_names}, f)

    await update.message.reply_text("✅ Yüz başarıyla misafir olarak eklendi.")

# Botu başlat
def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    print("🤖 Telegram botu dinlemede...")
    app.run_polling()

if __name__ == "__main__":
    main()
