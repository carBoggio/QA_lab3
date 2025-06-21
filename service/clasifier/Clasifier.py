from abc import ABC, abstractmethod
import numpy as np
from typing import List, Optional, Dict, Tuple


class BaseFaceClassifier(ABC):
    """
    Clase abstracta para clasificadores de reconocimiento facial.
    """

    def __init__(self):
        self.is_trained = False

    @abstractmethod
    def train(self, embeddings_dict: Dict[str, List[np.ndarray]]) -> bool:
        """
        Entrena el clasificador.

        Args:
            embeddings_dict: Diccionario con formato {"persona1": [embedding1, embedding2, ...], ...}

        Returns:
            bool: True si fue exitoso
        """
        pass

    @abstractmethod
    def predict(self, embedding: np.ndarray) -> Optional[Tuple[str, str]]:
        """
        Predice el estudiante para un embedding.

        Args:
            embedding: Embedding facial

        Returns:
            str: ID del estudiante o None
        """
        pass

    def batch_predict(self, embeddings: List[np.ndarray]) -> List[Optional[str]]:
        """
        Predice múltiples embeddings de una vez.

        Args:
            embeddings: Lista de embeddings faciales

        Returns:
            List[Optional[str]]: Lista de predicciones
        """
        predictions = []
        for embedding in embeddings:
            prediction = self.predict(embedding)
            predictions.append(prediction)
        return predictions
