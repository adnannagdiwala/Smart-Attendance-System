"""
Face Registration Script
Captures photos and creates embeddings for the face recognition system.
Works with face_recognition_system.py
"""

import cv2
import numpy as np
import torch
import os
from facenet_pytorch import InceptionResnetV1

# ============ CONFIGURATION ============
DATASET_FOLDER = "dataset"
EMBEDDINGS_FILE = "embeddings.npy"
NUM_PHOTOS = 5  # Number of photos to capture per person

# Face detector (same as recognition system)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Embedding model
model = InceptionResnetV1(pretrained='vggface2').eval()


def preprocess_face(face_img):
    """Preprocess face for InceptionResnetV1"""
    face_resized = cv2.resize(face_img, (160, 160))
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    face_tensor = torch.from_numpy(face_rgb).permute(2, 0, 1).float()
    face_tensor = (face_tensor - 127.5) / 128.0
    return face_tensor.unsqueeze(0)


def get_embedding(face_img):
    """Get face embedding"""
    face_tensor = preprocess_face(face_img)
    with torch.no_grad():
        embedding = model(face_tensor).numpy()[0]
    return embedding


def capture_faces(person_name):
    """Capture faces and save photos + embedding"""
    
    # Create person folder
    person_folder = os.path.join(DATASET_FOLDER, person_name)
    os.makedirs(person_folder, exist_ok=True)
    
    print(f"\n📸 Capturing {NUM_PHOTOS} photos for '{person_name}'")
    print("Position your face in the frame and press SPACE to capture")
    print("Press 'q' to cancel\n")
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    photos_captured = 0
    embeddings_list = []
    
    while photos_captured < NUM_PHOTOS:
        ret, frame = cap.read()
        if not ret:
            continue
        
        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(80, 80))
        
        face_detected = False
        face_img = None
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_detected = True
            face_img = frame[y:y+h, x:x+w]
            
            # Draw green box
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(display, "Face OK - Press SPACE", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(display, "No face detected", (30, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Show progress
        cv2.putText(display, f"Captured: {photos_captured}/{NUM_PHOTOS}", (30, 460),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, f"Person: {person_name}", (30, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Face Registration", display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("❌ Cancelled")
            cap.release()
            cv2.destroyAllWindows()
            return False
        
        if key == ord(' ') and face_detected:
            # Save photo
            photo_path = os.path.join(person_folder, f"{photos_captured + 1}.jpg")
            cv2.imwrite(photo_path, face_img)
            
            # Get embedding
            emb = get_embedding(face_img)
            embeddings_list.append(emb)
            
            photos_captured += 1
            print(f"   ✓ Photo {photos_captured}/{NUM_PHOTOS} captured")
            
            # Brief pause
            cv2.waitKey(300)
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Calculate average embedding for this person
    avg_embedding = np.mean(embeddings_list, axis=0)
    
    # Load or create embeddings database
    if os.path.exists(EMBEDDINGS_FILE):
        db = np.load(EMBEDDINGS_FILE, allow_pickle=True).item()
    else:
        db = {}
    
    # Add/update this person
    db[person_name] = avg_embedding
    
    # Save database
    np.save(EMBEDDINGS_FILE, db)
    
    print(f"\n✅ Successfully registered '{person_name}'!")
    print(f"   - {NUM_PHOTOS} photos saved to: {person_folder}")
    print(f"   - Embedding added to: {EMBEDDINGS_FILE}")
    print(f"   - Total people in database: {len(db)}")
    
    return True


def list_registered():
    """List all registered people"""
    if os.path.exists(EMBEDDINGS_FILE):
        db = np.load(EMBEDDINGS_FILE, allow_pickle=True).item()
        print(f"\n📋 Registered people ({len(db)}):")
        for name in db.keys():
            print(f"   - {name}")
    else:
        print("\n❌ No embeddings database found")


def delete_person(name):
    """Delete a person from the database"""
    if os.path.exists(EMBEDDINGS_FILE):
        db = np.load(EMBEDDINGS_FILE, allow_pickle=True).item()
        if name in db:
            del db[name]
            np.save(EMBEDDINGS_FILE, db)
            print(f"✅ Deleted '{name}' from database")
        else:
            print(f"❌ '{name}' not found in database")


# ============ MAIN ============
if __name__ == "__main__":
    os.makedirs(DATASET_FOLDER, exist_ok=True)
    
    print("=" * 50)
    print("    FACE REGISTRATION SYSTEM")
    print("=" * 50)
    
    while True:
        print("\nOptions:")
        print("  1. Register new person")
        print("  2. List registered people")
        print("  3. Delete person")
        print("  4. Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            name = input("Enter person's name: ").strip()
            if name:
                capture_faces(name)
            else:
                print("❌ Name cannot be empty")
        
        elif choice == "2":
            list_registered()
        
        elif choice == "3":
            name = input("Enter name to delete: ").strip()
            if name:
                delete_person(name)
        
        elif choice == "4":
            print("\nGoodbye!")
            break
        
        else:
            print("❌ Invalid choice")
