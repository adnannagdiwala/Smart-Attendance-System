from deepface import DeepFace
import cv2
import os
import time

print("=" * 60)
print("SMART ATTENDANCE - IMPROVED FACE RECOGNITION")
print("=" * 60)

# ============================================
# Load Face Database
# ============================================

def load_face_database():
    """Load all registered faces from dataset folder"""
    
    database = []
    dataset_path = "dataset"
    
    if not os.path.exists(dataset_path):
        print("❌ Dataset folder not found!")
        return []
    
    print("\n📂 Loading face database...")
    
    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)
        
        if not os.path.isdir(person_folder):
            continue
        
        print(f"   Loading {person_name}...", end=" ")
        
        image_files = [f for f in os.listdir(person_folder) if f.endswith('.jpg')]
        
        if len(image_files) == 0:
            print("⚠️  No images found")
            continue
        
        person_data = {
            'name': person_name,
            'images': []
        }
        
        for image_file in image_files:
            image_path = os.path.join(person_folder, image_file)
            person_data['images'].append(image_path)
        
        database.append(person_data)
        print(f"✓ {len(image_files)} images loaded")
    
    print(f"\n✓ Total people registered: {len(database)}")
    return database

# ============================================
# IMPROVED Recognition - Checks ALL people
# ============================================

def recognize_face(frame, database):
    """
    Check against ALL people and return BEST match
    """
    
    if len(database) == 0:
        return None
    
    all_matches = []  # Store ALL potential matches
    
    try:
        # Check EVERY person in database
        for person in database:
            person_name = person['name']
            
            print(f"   Checking against {person_name}...", end=" ")
            
            # Check first 3 images of this person
            distances = []
            
            for stored_image_path in person['images'][:3]:
                try:
                    result = DeepFace.verify(
                        img1_path=frame,
                        img2_path=stored_image_path,
                        model_name='Facenet',
                        detector_backend='opencv',
                        enforce_detection=False
                    )
                    
                    distances.append(result['distance'])
                
                except Exception as e:
                    continue
            
            # If we got any valid distances, use the minimum (best match)
            if len(distances) > 0:
                min_distance = min(distances)
                print(f"Distance: {min_distance:.4f}")
                
                # Store this person's best match
                all_matches.append({
                    'name': person_name,
                    'distance': min_distance
                })
            else:
                print("No valid comparison")
    
    except Exception as e:
        print(f"\nError during recognition: {e}")
    
    # Now find the BEST match (lowest distance)
    if len(all_matches) == 0:
        print("   ❌ No matches found")
        return None
    
    # Sort by distance (lowest first)
    all_matches.sort(key=lambda x: x['distance'])
    
    best_match = all_matches[0]
    
    # STRICT THRESHOLD - only accept if distance is low enough
    # For Facenet model, good matches are usually < 0.4
    THRESHOLD = 0.45
    
    if best_match['distance'] < THRESHOLD:
        # Get face location
        try:
            faces = DeepFace.extract_faces(
                img_path=frame,
                detector_backend='opencv',
                enforce_detection=False
            )
            
            if len(faces) > 0:
                confidence = max(0, min(100, (1 - best_match['distance']) * 100))
                
                print(f"\n   ✓✓✓ BEST MATCH: {best_match['name']} (Distance: {best_match['distance']:.4f}, Confidence: {confidence:.1f}%)")
                
                # Show runner-up if exists
                if len(all_matches) > 1:
                    print(f"   Runner-up: {all_matches[1]['name']} (Distance: {all_matches[1]['distance']:.4f})")
                
                return {
                    'name': best_match['name'],
                    'confidence': confidence,
                    'face_area': faces[0]['facial_area'],
                    'distance': best_match['distance']
                }
        except:
            pass
    else:
        print(f"   ❌ Best distance {best_match['distance']:.4f} exceeds threshold {THRESHOLD}")
    
    return None

# ============================================
# Main Recognition System
# ============================================

def start_recognition():
    """Start the face recognition camera system"""
    
    database = load_face_database()
    
    if len(database) == 0:
        print("\n❌ No registered faces found!")
        print("Please run 'capture_faces.py' first.")
        return
    
    print("\n" + "=" * 60)
    print("Starting camera...")
    print("Controls:")
    print("  - Press 'r' to recognize")
    print("  - Press 'q' to quit")
    print("=" * 60 + "\n")
    
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    time.sleep(2)
    
    last_recognized = None
    show_result_timer = 0
    
    print("Camera ready! Press 'r' to recognize face\n")
    
    while True:
        success, frame = camera.read()
        
        if not success:
            print("❌ Failed to capture frame")
            break
        
        display_frame = frame.copy()
        
        # Display last recognized person
        if last_recognized and show_result_timer > 0:
            face = last_recognized['face_area']
            x, y, w, h = face['x'], face['y'], face['w'], face['h']
            
            # Draw green rectangle
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
            
            # Display name and confidence
            name = last_recognized['name']
            conf = last_recognized['confidence']
            dist = last_recognized['distance']
            
            label = f"{name} ({conf:.1f}%)"
            dist_label = f"Dist: {dist:.4f}"
            
            # Background for text
            cv2.rectangle(display_frame, (x, y-60), (x+w, y), (0, 255, 0), -1)
            cv2.putText(display_frame, label, (x+5, y-35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(display_frame, dist_label, (x+5, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            show_result_timer -= 1
        
        # Display instructions
        cv2.putText(display_frame, "Press 'r' to recognize | 'q' to quit", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow("Smart Attendance - Face Recognition", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\nStopping...")
            break
        
        elif key == ord('r'):
            print("\n🔍 Recognizing... (checking all people)")
            print("-" * 60)
            
            recognized = recognize_face(frame, database)
            
            if recognized:
                last_recognized = recognized
                show_result_timer = 90  # Show for 3 seconds
            else:
                last_recognized = None
                print("   No confident match found")
            
            print("-" * 60 + "\n")
    
    camera.release()
    cv2.destroyAllWindows()
    print("✓ Recognition system stopped")

# ============================================
# Run
# ============================================

if __name__ == "__main__":
    start_recognition()