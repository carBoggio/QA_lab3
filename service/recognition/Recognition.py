from abc import ABC, abstractmethod
from PIL import Image
import numpy as np
from typing import List


class Recognition(ABC):
    @abstractmethod
    def generate_embedding(self, face_image: Image.Image) -> np.ndarray:
        """
        Generate an embedding vector for a cropped face image.

        Args:
            face_image: PIL Image containing a cropped face

        Returns:
            numpy array containing the face embedding vector
        """
        pass

    @abstractmethod
    def generate_embeddings(self, face_images: List[Image.Image]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple face images.

        Args:
            face_images: List of PIL Images containing cropped faces

        Returns:
            List of numpy arrays containing the face embedding vectors
        """
        pass

    @abstractmethod
    def predict(self, embedding: np.ndarray) -> float:
        """
        Compute similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Similarity score between the embeddings
        """
        pass
