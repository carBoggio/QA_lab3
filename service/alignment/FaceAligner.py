from typing import Tuple, Optional, Union, List
from abc import ABC, abstractmethod
from PIL import Image
import numpy as np

class FaceAligner(ABC):
    """
    Clase abstracta base para alineadores de caras.
    
    Define la interfaz común que deben implementar todos los alineadores de caras.
    Permite tener diferentes estrategias de alineación (afín, proyectiva, etc.)
    mientras manteniendo una interfaz consistente.
    """
    
    def __init__(self, target_size: Tuple[int, int] = (112, 112)):
        """
        Inicializa el alineador base.
        
        Args:
            target_size: Tamaño objetivo (width, height) para las caras alineadas
        """
        self.target_size = target_size
    
    @abstractmethod
    def align(self, image: Union[Image.Image, np.ndarray]) -> Image.Image:
        """
        Alinea una cara en la imagen.
        
        Args:
            image: Imagen PIL o array numpy con una cara
            
        Returns:
            Imagen PIL con la cara alineada
        """
        pass
    