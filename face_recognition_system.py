import cv2, numpy as np, torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from scipy.spatial.distance import cosine

db = np.load("embeddings.npy", allow_pickle=True).item()

mtcnn = MTCNN(image_size=160, margin=20)
model = InceptionResnetV1(pretrained='vggface2').eval()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face = mtcnn(rgb)

    name = "Unknown"
    if face is not None:
        emb = model(face.unsqueeze(0)).detach().numpy()[0]
        min_dist = 1

        for person, ref in db.items():
            dist = cosine(emb, ref)
            if dist < min_dist:
                min_dist = dist
                name = person

        if min_dist > 0.6:
            name = "Unknown"

    cv2.putText(frame, name, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0) if name != "Unknown" else (0, 0, 255), 2)

    cv2.imshow("Face Attendance", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
