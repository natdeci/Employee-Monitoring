import os
import cv2
from insightface import app


class FaceDatabase:
    def __init__(self, database_path):
        self.database_path = database_path
        self.face_embeddings = self._load_database()

    def _load_database(self):
        embeddings = {}
        face_recognition = app.FaceAnalysis()
        face_recognition.prepare(ctx_id=0, det_size=(640, 640))
        for person_name in os.listdir(self.database_path):
            person_folder = os.path.join(self.database_path, person_name)
            if os.path.isdir(person_folder):
                embeddings[person_name] = []
                for img_name in os.listdir(person_folder):
                    img_path = os.path.join(person_folder, img_name)
                    img = cv2.imread(img_path)
                    if img is not None:
                        faces = face_recognition.get(img)
                        for face in faces:
                            embeddings[person_name].append(face.normed_embedding)
        return embeddings
