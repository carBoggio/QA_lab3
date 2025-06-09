from abc import ABC, abstractmethod
from PIL import Image
import numpy as np
from typing import List
from deepface import DeepFace
from .Recognition import Recognition
from ..clasifier.clasifier import KNNFaceClassifier
from torchvision import transforms

class ArcFaceDeepFaceRecognition(Recognition):
    def __init__(self, model_name: str = 'ArcFace'):
        self.model_name = model_name
        self.knn_classifier = KNNFaceClassifier()
        self.epsilon = 1e-10  # Pequeño valor para evitar división por cero
        self.transform = transforms.Compose([
            transforms.Resize(112, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(112),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.numpy() * 255)
        ])
    def generate_embedding(self, face_image: Image.Image) -> np.ndarray:
        """
        Generate an embedding vector for a cropped face image.
        """
        try:
            face_image = face_image.resize((112, 112), Image.LANCZOS)
            face_array = np.array(face_image)
            
            result = DeepFace.represent(
                img_path=face_array,
                model_name=self.model_name,
                enforce_detection=False
            )
            
            if isinstance(result, list):
                embedding = np.array(result[0]['embedding'])
            else:
                embedding = np.array(result['embedding'])
            
            return self._normalize_l2(embedding)
            
        except Exception as e:
            print(f"Error generando embedding: {e}")
            return np.array([])

    def generate_embeddings(self, face_images: List[Image.Image]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple face images.
        """
        embeddings = []
        for face_image in face_images:
            embedding = self.generate_embedding(face_image)
            if embedding.size > 0:
                embeddings.append(embedding)
        return embeddings

    def predict(self, embedding: np.ndarray) -> float:
        """
        Compute similarity between two embeddings using cosine similarity.
        """
        # Normalizar embeddings antes de comparar
        emb_normalized = self._normalize_l2(embedding)
        
        prediction = self.knn_classifier.predict(emb_normalized)
        
        return prediction

    def _normalize_l2(self, embedding: np.ndarray) -> np.ndarray:
        """
        Normalización L2 del embedding
        """
        norm = np.linalg.norm(embedding)
        
        return embedding / (norm + self.epsilon)