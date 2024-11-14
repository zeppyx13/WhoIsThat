import cv2
import mysql.connector
from Logic_Recognition import SimpleFacerec
import threading
import time
import os

# Konfigurasi koneksi ke database MySQL
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'face_recognition'
}

# Variabel untuk menyimpan waktu deteksi terakhir dari setiap nama
last_detection_time = {}

# Path untuk menyimpan foto yang diambil
detected_folder = "detected"
os.makedirs(detected_folder, exist_ok=True)


def save_detection_to_db(name, location=None, confidence_level=None):
    try:
        # Simpan semua data jika nama adalah 'Unknown' atau jika confidence_level > 0.5 untuk wajah yang dikenal
        if name == 'Unknown' or (confidence_level is not None and confidence_level > 0.5):
            confidence_level = float(confidence_level) if confidence_level is not None else None

            # Proses penyimpanan ke database
            conn = mysql.connector.connect(**db_config)
            cursor = conn.cursor()
            query = '''
                INSERT INTO detected_faces (name, location, confidence_level)
                VALUES (%s, %s, %s)
            '''
            cursor.execute(query, (name, location, confidence_level))
            conn.commit()
            cursor.close()
            conn.close()
    except mysql.connector.Error as err:
        print(f"Error: {err}")


def save_image(frame, name):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{detected_folder}/{name}_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    print(f"Image saved: {filename}")


# Encode wajah dari folder
sfr = SimpleFacerec()
sfr.load_encoding_images("Member/")

# Load kamera
cap = cv2.VideoCapture(0)

# Set resolusi kamera agar performa lebih baik
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


def process_frame():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame agar tidak mirror
        frame = cv2.flip(frame, 1)

        # Deteksi wajah
        face_locations, face_names, confidences = sfr.detect_known_faces(frame)

        # Menampilkan hasil deteksi
        current_time = time.time()
        for face_loc, name, confidence in zip(face_locations, face_names, confidences):
            y1, x2, y2, x1 = face_loc

            # Pastikan wajah tidak terlalu dekat dengan kamera
            if (x2 - x1) > 100 and (y2 - y1) > 100:  # Cek ukuran minimal wajah
                cv2.putText(frame, f"{name} ({confidence:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200),
                            2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

                # Jika confidence level di atas 0.5 dan nama belum terdeteksi dalam 10 detik terakhir, simpan foto
                if confidence > 0.5 and (
                        name not in last_detection_time or (current_time - last_detection_time[name] > 10)):
                    save_detection_to_db(name, 'Camera 1', confidence)
                    save_image(frame, name)
                    last_detection_time[name] = current_time  # Perbarui waktu deteksi terakhir

        # Tampilkan frame
        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1)
        if key == 27:  # Tekan 'Esc' untuk keluar
            break


# Jalankan pemrosesan frame di thread terpisah
thread = threading.Thread(target=process_frame)
thread.start()

thread.join()
cap.release()
cv2.destroyAllWindows()
