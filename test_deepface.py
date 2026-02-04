print("Testing DeepFace installation...")

try:
    from deepface import DeepFace
    import cv2
    print("✓ All libraries imported successfully!")
    print("✓ DeepFace is ready to use!")
    
    # Test camera
    camera = cv2.VideoCapture(0)
    if camera.isOpened():
        print("✓ Camera is working!")
        camera.release()
    else:
        print("✗ Camera not detected")
        
except ImportError as e:
    print(f"✗ Import error: {e}")
except Exception as e:
    print(f"✗ Error: {e}")