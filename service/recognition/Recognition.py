from abc import ABC, abstractmethod
from PIL import Image
import numpy as np
from typing import List, Optional


class Recognition(ABC):
    @abstractmethod
    def generate_embedding(self, face_array: np.ndarray) -> np.ndarray:
        """
        Generate an embedding vector for a cropped face image.

        Args:
            face_array: numpy array containing a cropped face

        Returns:
            numpy array containing the face embedding vector
        """
        pass

    @abstractmethod
    def generate_embeddings(self, face_arrays: List[np.ndarray]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple face images.

        Args:
            face_arrays: List of numpy arrays containing cropped faces

        Returns:
            List of numpy arrays containing the face embedding vectors
        """
        pass
