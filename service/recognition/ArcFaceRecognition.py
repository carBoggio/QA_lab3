from abc import ABC, abstractmethod
from PIL import Image
import numpy as np
from typing import List
from deepface import DeepFace
from .Recognition import Recognition
import cv2

from ..utils import resize_with_padding


class ArcFaceDeepFaceRecognition(Recognition):
    def __init__(self, model_name: str = "ArcFace"):
        self.model_name = model_name

        self.epsilon = 1e-10

    def generate_embedding(self, face_array: np.ndarray) -> np.ndarray:

        try:
            # La imagen ya está recortada y normalizada, por lo que nos saltamos la detección
            # de DeepFace para evitar errores y usar nuestra normalización superior.
            result = DeepFace.represent(
                img_path=face_array,
                model_name="ArcFace",
                detector_backend="skip",  # No detectar, usar la imagen tal cual
                align=False,              # No alinear, ya lo hicimos con STAR Loss
                normalization="ArcFace",
            )

            if isinstance(result, list) and len(result) > 0:
                embedding = np.array(result[0]["embedding"])
            elif isinstance(result, dict) and "embedding" in result:
                embedding = np.array(result["embedding"])
            else:
                embedding = np.array([])

            return embedding

        except Exception as e:
            print(f"Error generando embedding: {e}")
            return np.array([])

    def generate_embeddings(self, face_arrays: List[np.ndarray]) -> List[np.ndarray]:

        embeddings = []
        for face_array in face_arrays:
            embedding = self.generate_embedding(face_array)
            if embedding.size > 0:
                embeddings.append(embedding)
        return embeddings
