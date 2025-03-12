import os
import cv2
import numpy as np
import platform
import onnxruntime as ort
from insightface.app import FaceAnalysis

class FaceRecognizer:
    def __init__(self, database_path="family_photos"):
        # Configure provider based on platform
        if platform.processor() == 'arm':
            # For Apple Silicon (M1/M2/M3)
            providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        self.app = FaceAnalysis(
            name='buffalo_l',
            providers=providers,
            allowed_modules=['detection', 'recognition']
        )
        
        # Configure for Apple Silicon
        self.app.prepare(
            ctx_id=0,  # Use 0 for Apple Silicon
            det_size=(640, 640)
        )
        
        self.database_path = database_path
        self.known_embeddings = {}
        self.recognition_threshold = 0.55
        print(f"Initializing ArcFace recognition system using {providers[0]}...")
        self._load_known_faces()
    
    def _load_known_faces(self):
        if not os.path.exists(self.database_path):
            return
            
        for family_name in os.listdir(self.database_path):
            family_path = os.path.join(self.database_path, family_name)
            if os.path.isdir(family_path):
                # Process each face image in family folder
                for img_name in os.listdir(family_path):
                    if img_name.endswith('_face.jpg'):
                        img_path = os.path.join(family_path, img_name)
                        person_name = os.path.splitext(img_name)[0][:-5]
                        img = cv2.imread(img_path)
                        faces = self.app.get(img)
                        if faces:
                            self.known_embeddings[person_name] = {
                                'embedding': faces[0].embedding,
                                'family': family_name
                            }
    
    def identify_face(self, frame):
        """Identify face in frame"""
        faces = self.app.get(frame)
        if not faces:
            return None, None, 0
            
        face_embedding = faces[0].embedding
        
        if not self.known_embeddings:
            return None, None, 0
            
        best_match = None
        best_family = None
        highest_similarity = 0
        
        # Compare embeddings
        norm_embedding = face_embedding / np.linalg.norm(face_embedding)
        for person_name, data in self.known_embeddings.items():
            known_embedding = data['embedding']
            norm_known = known_embedding / np.linalg.norm(known_embedding)
            similarity = np.dot(norm_embedding, norm_known)
            
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = person_name
                best_family = data['family']
        
        if highest_similarity > self.recognition_threshold:
            return best_match, best_family, highest_similarity
            
        return None, None, highest_similarity
