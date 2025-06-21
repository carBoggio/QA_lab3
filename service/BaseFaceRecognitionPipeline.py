from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union
import numpy as np
from PIL import Image


class BaseFaceRecognitionPipeline(ABC):
    """
    Clase abstracta que define la interfaz para pipelines de reconocimiento facial.

    Un pipeline completo incluye:
    1. Detección de caras en imágenes
    2. Extracción de características (embeddings)
    3. Clasificación/Identificación de personas
    """


    @abstractmethod
    def extract_embedding_from_single_largest_face_image(
        self, face_image: Image.Image
    ) -> Optional[np.ndarray]:
        """
        Genera un vector de características

        (embedding) a partir
        de una imagen que contiene UNA SOLA cara, si hay varias, usa la mas grande.

        Args:
            face_image: Imagen PIL que contiene exactamente una cara (ya recortada)

        Returns:
            np.ndarray: Vector de características facial o None si hay error
        """
        pass

    @abstractmethod
    def predict_people_identity_from_picture(
        self, picture: Image.Image
    ) -> List[Tuple[str, str]]:
        """
        Predice la identidad de todas las personas en una foto.

        Args:
            picture: Imagen PIL que contiene todas las personas que se quieren identificar

        Returns:
            List[Tuple[str, str]]: Lista de tuplas con el nombre de la persona y el nivel de confianza
                - "presente" - se detecta una persona pero no se puede identificar específicamente
                - "no_identificado" - no se puede clasificar la persona
        """
        pass

    @abstractmethod
    def draw_faces_in_picture(
        self, picture: Image.Image
    ) -> Image.Image:
        """
        Dibuja las caras en una imagen.
        """
        pass

