from deepface import DeepFace
import cv2
import time

print("Starting Face Detection...")
print("Press 'q' to quit")

# Initialize camera
camera = cv2.VideoCapture(0)

# Give camera time to warm up
time.sleep(1)

while True:
    # Capture frame
    success, frame = camera.read()
    
    if not success:
        print("Failed to capture frame")
        break
    
    try:
        # Detect faces in the frame
        result = DeepFace.extract_faces(
            frame, 
            detector_backend='opencv',
            enforce_detection=False
        )
        
        # Draw rectangles around detected faces
        for face in result:
            # Get face coordinates
            x = face['facial_area']['x']
            y = face['facial_area']['y']
            w = face['facial_area']['w']
            h = face['facial_area']['h']
            
            # Draw green rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Add text label
            cv2.putText(frame, "Face Detected", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
    except Exception as e:
        # If no face detected or error, just continue
        pass
    
    # Display the frame
    cv2.imshow("Smart Attendance - Face Detection", frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
camera.release()
cv2.destroyAllWindows()
print("Face detection stopped")