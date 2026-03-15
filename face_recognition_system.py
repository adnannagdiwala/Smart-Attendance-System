import cv2
import numpy as np
import torch
import time
from facenet_pytorch import InceptionResnetV1
from scipy.spatial.distance import cosine
from threading import Thread
from attendance_db import init_db, log_attendance

# ============ THREADED VIDEO CAPTURE ============
class VideoCapture:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.ret, self.frame = self.cap.read()
        self.running = True
        Thread(target=self._update, daemon=True).start()

    def _update(self):
        while self.running:
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.ret, self.frame.copy() if self.frame is not None else None

    def release(self):
        self.running = False
        self.cap.release()


# ============ CONFIGURATION ============
PROCESS_EVERY_N_FRAMES = 8   # Process every 8th frame
THRESHOLD = 0.6

# Use OpenCV's fast Haar Cascade (MUCH faster than MTCNN on CPU)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load recognition model
model = InceptionResnetV1(pretrained='vggface2').eval()
db = np.load("embeddings.npy", allow_pickle=True).item()

print("Loaded database with", len(db), "people")

# Initialize attendance database
init_db()


def preprocess_face(face_img):
    """Preprocess face for InceptionResnetV1"""
    face_resized = cv2.resize(face_img, (160, 160))
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    face_tensor = torch.from_numpy(face_rgb).permute(2, 0, 1).float()
    face_tensor = (face_tensor - 127.5) / 128.0  # Normalize
    return face_tensor.unsqueeze(0)


# ============ MAIN LOOP ============
cap = VideoCapture(0)
frame_count = 0
cached_name = "Unknown"
cached_box = None
logged_indicator_until = 0  # timestamp until which to show "✓ Logged" on screen

print("Press ESC to exit...")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    frame_count += 1

    # Only process every Nth frame
    if frame_count % PROCESS_EVERY_N_FRAMES == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Fast face detection
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(80, 80))

        name = "Unknown"
        box = None

        if len(faces) > 0:
            x, y, w, h = faces[0]  # Take first face
            box = (x, y, w, h)
            
            # Extract and recognize face
            face_img = frame[y:y+h, x:x+w]
            face_tensor = preprocess_face(face_img)

            with torch.no_grad():
                emb = model(face_tensor).numpy()[0]

            min_dist = 1
            for person, ref in db.items():
                dist = cosine(emb, ref)
                if dist < min_dist:
                    min_dist = dist
                    name = person

            if min_dist > THRESHOLD:
                name = "Unknown"

        # Log attendance for recognized faces
        if name != "Unknown":
            if log_attendance(name):
                logged_indicator_until = time.time() + 2  # show indicator for 2 seconds

        cached_name = name
        cached_box = box

    # Draw cached result
    if cached_box:
        x, y, w, h = cached_box
        color = (0, 255, 0) if cached_name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, cached_name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    else:
        cv2.putText(frame, cached_name, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show "Logged" indicator briefly after attendance is recorded
    if time.time() < logged_indicator_until:
        cv2.putText(frame, "Attendance Logged", (30, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Face Attendance", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
