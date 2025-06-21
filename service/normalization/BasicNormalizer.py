from abc import ABC, abstractmethod
import numpy as np
from typing import List, Union
from PIL import Image


class BasicNormalizer(ABC):
    """
    Clase abstracta para normalización de rostros.
    Define la interfaz que deben implementar todas las clases de normalización.
    """

    @abstractmethod
    def normaliceFace(self, imagen: Image.Image) -> Image.Image:
        """
        Método abstracto para normalizar un rostro en una imagen.

        Args:
            imagen: Imagen PIL (Pillow)

        Returns:
            Image.Image: Imagen normalizada como objeto PIL
        """
        pass

    def normaliceFaces(self, imagenes: List[Image.Image]) -> List[Image.Image]:
        """
        Método para normalizar múltiples rostros en una lista de imágenes.

        Args:
            imagenes: Lista de imágenes PIL (Pillow)

        Returns:
            List[Image.Image]: Lista de imágenes normalizadas como objetos PIL
        """
        normalized_faces = []
        for i, imagen in enumerate(imagenes):
            try:
                normalized_face = self.normaliceFace(imagen)
                normalized_faces.append(normalized_face)
            except Exception as e:
                print(f"Error normalizando cara {i}: {str(e)}")
                # Continuar con la siguiente imagen en lugar de fallar todo el proceso
                continue
        return normalized_faces