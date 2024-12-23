from scipy.spatial.distance import cosine
from insightface import app


class FaceRecognitionSystem:
    def __init__(self, face_database):
        self.face_database = face_database
        self.recognition = app.FaceAnalysis()
        self.recognition.prepare(ctx_id=0, det_size=(640, 640))

    def recognize(self, face_image):
        faces = self.recognition.get(face_image)
        if not faces:
            return None, "Unknown"
        face = faces[0]
        identity = face.normed_embedding

        min_distance = float('inf')
        recognized_person = None
        for person_name, embeddings in self.face_database.face_embeddings.items():
            for embedding in embeddings:
                distance = cosine(identity, embedding)
                if distance < min_distance:
                    min_distance = distance
                    recognized_person = person_name

        if min_distance < 0.8:  # Adjust threshold as needed
            return face.bbox.astype(int), recognized_person
        return face.bbox.astype(int), "Unknown"