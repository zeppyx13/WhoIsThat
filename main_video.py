import cv2
import mysql.connector
from simple_facerec import SimpleFacerec
import threading

# Konfigurasi koneksi ke database MySQL
db_config = {
    'host': 'localhost',
    'user': 'root',  # Sesuaikan dengan user MySQL Anda
    'password': '',  # Ganti dengan password MySQL Anda
    'database': 'face_recognition'
}

# Fungsi untuk menyimpan data deteksi ke database
def save_detection_to_db(name, location=None, confidence_level=None):
    try:
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
        face_locations, face_names = sfr.detect_known_faces(frame)

        # Menampilkan hasil deteksi
        for face_loc, name in zip(face_locations, face_names):
            y1, x2, y2, x1 = face_loc

            # Pastikan wajah tidak terlalu dekat dengan kamera
            if (x2 - x1) > 100 and (y2 - y1) > 100:  # Cek ukuran minimal wajah
                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

                # Simpan ke database jika wajah terdeteksi
                save_detection_to_db(name, 'Camera 1')

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
