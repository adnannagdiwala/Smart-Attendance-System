import os
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

mtcnn = MTCNN(image_size=160, margin=20)
model = InceptionResnetV1(pretrained='vggface2').eval()

database = {}

for person in os.listdir("dataset"):
    embs = []
    person_path = os.path.join("dataset", person)

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face = mtcnn(img)

        if face is not None:
            emb = model(face.unsqueeze(0)).detach().numpy().flatten()
            embs.append(emb)

    if len(embs) > 0:
        database[person] = np.mean(embs, axis=0)

np.save("embeddings.npy", database)
print("Embeddings saved successfully")

