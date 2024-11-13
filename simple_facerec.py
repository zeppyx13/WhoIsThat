import face_recognition
import cv2
import os
import glob
import numpy as np

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

        # Resize frame for a faster speed (can adjust as needed)
        self.frame_resizing = 0.25

    def load_encoding_images(self, images_path):
        """
        Load encoding images from path.
        :param images_path: Path to the folder containing images.
        """
        # Load Images
        images_path = glob.glob(os.path.join(images_path, "*.*"))
        print(f"{len(images_path)} encoding images found.")

        # Store image encoding and names
        for img_path in images_path:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Get the filename only from the initial file path.
            basename = os.path.basename(img_path)
            filename, ext = os.path.splitext(basename)

            # Get encoding, skip if no face found.
            encodings = face_recognition.face_encodings(rgb_img)
            if encodings:
                img_encoding = encodings[0]
                # Store file name and file encoding
                self.known_face_encodings.append(img_encoding)
                self.known_face_names.append(filename)
            else:
                print(f"Warning: No face detected in {img_path}. Skipping this image.")

        print("Encoding images loaded")

    def detect_known_faces(self, frame, model='hog'):
        # hog is faster but less accurate (CPU), cnn is slower but more accurate (GPU)
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing,
                                 interpolation=cv2.INTER_LINEAR)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Deteksi wajah dalam frame
        face_locations = face_recognition.face_locations(rgb_small_frame, model=model)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        confidences = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            name = "Unknown"
            confidence = None

            if len(face_distances) > 0:
                confidence = 1 - min(face_distances)  # Confidence dihitung dari jarak terendah

            if matches:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]

            face_names.append(name)
            confidences.append(confidence)

        face_locations = np.array(face_locations) / self.frame_resizing
        return face_locations.astype(int), face_names, confidences
