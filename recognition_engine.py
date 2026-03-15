"""
Recognition Engine — Reusable face recognition for the web dashboard.
Handles webcam streaming, face detection/recognition, and attendance logging.
"""

import cv2
import numpy as np
import torch
import time
import os
from threading import Thread, Lock
from facenet_pytorch import InceptionResnetV1
from scipy.spatial.distance import cosine
from attendance_db import log_attendance

# ============ CONFIGURATION ============
EMBEDDINGS_FILE = "embeddings.npy"
DATASET_FOLDER = "dataset"
THRESHOLD = 0.6
PROCESS_EVERY_N_FRAMES = 6


class RecognitionEngine:
    """Loads the FaceNet model and embeddings, provides face recognition."""

    def __init__(self):
        self.model = InceptionResnetV1(pretrained='vggface2').eval()
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.db = {}
        self.reload_embeddings()

    def reload_embeddings(self):
        """Reload embeddings from file."""
        if os.path.exists(EMBEDDINGS_FILE):
            self.db = np.load(EMBEDDINGS_FILE, allow_pickle=True).item()
        else:
            self.db = {}

    def preprocess_face(self, face_img):
        """Preprocess face for InceptionResnetV1."""
        face_resized = cv2.resize(face_img, (160, 160))
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_tensor = torch.from_numpy(face_rgb).permute(2, 0, 1).float()
        face_tensor = (face_tensor - 127.5) / 128.0
        return face_tensor.unsqueeze(0)

    def detect_faces(self, frame):
        """Detect faces in a frame. Returns list of (x, y, w, h)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(80, 80))
        return faces

    def recognize_face(self, face_img):
        """Recognize a face image. Returns (name, distance)."""
        face_tensor = self.preprocess_face(face_img)
        with torch.no_grad():
            emb = self.model(face_tensor).numpy()[0]

        min_dist = 1.0
        best_name = "Unknown"
        for person, ref in self.db.items():
            dist = cosine(emb, ref)
            if dist < min_dist:
                min_dist = dist
                best_name = person

        if min_dist > THRESHOLD:
            best_name = "Unknown"

        return best_name, min_dist

    def get_embedding(self, face_img):
        """Get embedding vector for a face image."""
        face_tensor = self.preprocess_face(face_img)
        with torch.no_grad():
            emb = self.model(face_tensor).numpy()[0]
        return emb


class CameraStream:
    """Thread-safe webcam capture."""

    def __init__(self, src=0):
        self.cap = None
        self.src = src
        self.running = False
        self.frame = None
        self.lock = Lock()

    def start(self):
        """Open the camera and start capturing."""
        if self.running:
            return True

        self.cap = cv2.VideoCapture(self.src)
        if not self.cap.isOpened():
            return False

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.running = True

        ret, frame = self.cap.read()
        if ret:
            with self.lock:
                self.frame = frame

        Thread(target=self._update, daemon=True).start()
        return True

    def _update(self):
        """Continuously read frames in background thread."""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            time.sleep(0.01)

    def read(self):
        """Get the latest frame (thread-safe)."""
        with self.lock:
            if self.frame is not None:
                return True, self.frame.copy()
            return False, None

    def stop(self):
        """Release the camera."""
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.frame = None

    @property
    def is_active(self):
        return self.running


# ============ GLOBAL INSTANCES ============
engine = RecognitionEngine()
camera = CameraStream()

# Track recently logged names for the live feed overlay
recent_logs = []  # list of {"name": ..., "time": ..., "until": timestamp}


def generate_attendance_frames():
    """
    Generator that yields MJPEG frames with face recognition overlay.
    Auto-logs attendance for recognized faces.
    """
    global recent_logs

    if not camera.start():
        # Yield a "no camera" placeholder frame
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Camera not available", (120, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        _, buf = cv2.imencode('.jpg', placeholder)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        return

    # Reload embeddings each time streaming starts
    engine.reload_embeddings()

    frame_count = 0
    cached_results = []  # list of {"box": (x,y,w,h), "name": str}

    while camera.is_active:
        ret, frame = camera.read()
        if not ret or frame is None:
            time.sleep(0.03)
            continue

        frame_count += 1

        # Process every Nth frame for performance
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            faces = engine.detect_faces(frame)
            results = []

            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                name, dist = engine.recognize_face(face_img)
                results.append({"box": (x, y, w, h), "name": name})

                # Log attendance
                if name != "Unknown":
                    if log_attendance(name):
                        recent_logs.append({
                            "name": name,
                            "time": time.strftime("%H:%M:%S"),
                            "until": time.time() + 3,
                        })

            cached_results = results

        # Draw results on frame
        now = time.time()
        for r in cached_results:
            x, y, w, h = r["box"]
            name = r["name"]
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            # Name label with background
            label = name
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x, y - th - 10), (x + tw, y), color, -1)
            cv2.putText(frame, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show recent attendance logs on frame
        recent_logs = [l for l in recent_logs if now < l["until"]]
        y_offset = 30
        for log in recent_logs[-3:]:
            text = f"Logged: {log['name']} at {log['time']}"
            cv2.putText(frame, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            y_offset += 25

        # Status bar at bottom
        people_count = len(engine.db)
        status = f"Tracking {people_count} people | Faces: {len(cached_results)}"
        cv2.putText(frame, status, (10, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Encode and yield
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

        time.sleep(0.03)  # ~30 FPS cap


def generate_register_frames():
    """
    Generator that yields MJPEG frames with face detection overlay (no recognition).
    Used during webcam registration.
    """
    if not camera.start():
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Camera not available", (120, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        _, buf = cv2.imencode('.jpg', placeholder)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        return

    frame_count = 0
    cached_faces = []

    while camera.is_active:
        ret, frame = camera.read()
        if not ret or frame is None:
            time.sleep(0.03)
            continue

        frame_count += 1

        if frame_count % 4 == 0:
            cached_faces = engine.detect_faces(frame)

        # Draw face boxes
        for (x, y, w, h) in cached_faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Face OK", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if len(cached_faces) == 0:
            cv2.putText(frame, "No face detected", (180, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.putText(frame, "Position face and click Capture", (10, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

        time.sleep(0.03)


def capture_current_frame(person_name, photo_index):
    """
    Capture a single frame from the camera, save it, and return the embedding.
    Returns (success, message, embedding_or_None).
    """
    ret, frame = camera.read()
    if not ret or frame is None:
        return False, "Camera not available.", None

    faces = engine.detect_faces(frame)
    if len(faces) == 0:
        return False, "No face detected. Try again.", None

    x, y, w, h = faces[0]
    face_img = frame[y:y+h, x:x+w]

    # Save photo
    person_folder = os.path.join(DATASET_FOLDER, person_name)
    os.makedirs(person_folder, exist_ok=True)
    photo_path = os.path.join(person_folder, f"{photo_index}.jpg")
    cv2.imwrite(photo_path, face_img)

    # Get embedding
    emb = engine.get_embedding(face_img)

    return True, f"Photo {photo_index} captured.", emb
