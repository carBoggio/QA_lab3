from abc import ABC, abstractmethod
from typing import List, Tuple
from PIL import Image


class Detector(ABC):
    """
    Clase abstracta base para detectores de rostros
    """

    @abstractmethod
    def detect_faces(self, image_file: Image.Image) -> List[Tuple[int, int, int, int]]:
        """
        Detecta rostros en una imagen y retorna sus coordenadas

        Args:
            image_file: Imagen a procesar

        Returns:
            Lista de tuplas con las coordenadas (x1, y1, x2, y2) de cada rostro
        """
        pass

    @abstractmethod
    def get_faces_choped_and_aligned(
        self, image_file: Image.Image
    ) -> List[Image.Image]:
        """
        Extrae caras recortadas y alineadas de la imagen

        Args:
            image_file: Imagen a procesar

        Returns:
            Lista de imágenes PIL con las caras recortadas y alineadas
        """
        pass
