import cv2
from simple_facerec import SimpleFacerec
import threading

# Encode faces from a folder
sfr = SimpleFacerec()
sfr.load_encoding_images("Member/")

# Load Camera
cap = cv2.VideoCapture(0)

# Set proper resolution for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def process_frame():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Reduce frame size for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect Faces
        face_locations, face_names = sfr.detect_known_faces(rgb_small_frame)

        # Scale back up face locations since the frame we detected in was scaled to 1/2 size
        for face_loc, name in zip(face_locations, face_names):
            y1, x2, y2, x1 = [int(coord * 2) for coord in face_loc]

            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

        # Display the resulting frame
        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1)
        if key == 27:  # Press 'Esc' to exit
            break

# Run frame processing in a separate thread to avoid blocking
thread = threading.Thread(target=process_frame)
thread.start()

thread.join()
cap.release()
cv2.destroyAllWindows()
