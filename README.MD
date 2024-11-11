# Simple Face Recognition with OpenCV and Face Recognition Library

![Python](https://img.shields.io/badge/Python-3.x-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![Face Recognition](https://img.shields.io/badge/face--recognition-1.3.0-yellow)

## Deskripsi Proyek

Proyek ini adalah sistem deteksi dan pengenalan wajah yang sederhana menggunakan Python, OpenCV, dan pustaka `face_recognition`. Proyek ini memuat gambar wajah yang dikodekan dari folder tertentu dan kemudian menggunakan kamera untuk mendeteksi dan mencocokkan wajah yang terdeteksi dengan data wajah yang sudah dikenal.

## Fitur

- **Pengenalan wajah secara real-time** melalui webcam.
- **Pendeteksian wajah yang akurat** menggunakan algoritma `HOG` dan `CNN`.
- **Membaca dan menyimpan encoding wajah** dari folder untuk mengidentifikasi wajah yang dikenal.
- **Penanganan wajah yang tidak dikenal** dengan penanda "Unknown".

## Prasyarat

Pastikan Anda sudah menginstal dependensi berikut:

- Python 3.6 atau lebih tinggi
- OpenCV
- Dlib
- face_recognition
