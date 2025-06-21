from abc import ABC
import dlib
from PIL import Image
from .BasicNormalizer import BasicNormalizer
import numpy as np
import cv2
import os

class DlibFaceNormalizer(BasicNormalizer):
    # Variable de clase para las rutas de la máscara facial (se inicializa una vez)
    FACE_MASK_ROUTES = routes = [i for i in range(16, -1, -1)] + [i for i in range(17,19)] + [i for i in range(24,26)] + [16]
    
    def __init__(self):
        # Modelos๋࣭ ⭑💃📷👸🏻.𖥔 ݁ ˖
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "shape_predictor_68_face_landmarks.dat")

        print(f"Buscando modelo en: {model_path}")
        self.landmarks_detector = dlib.shape_predictor(model_path)
    
    """
    Implementación de normalización de rostros usando dlib.
    Encuentra landmarks faciales y normaliza la orientación y tamaño.
    input: imagen de tipo PIL.Image con una sola cara ya recortada (asume que ya fue detectada por RetinaFace)
    """

    def _gray(self, np_array: np.ndarray) -> np.ndarray:
        if len(np_array.shape) == 3:
            return cv2.cvtColor(np_array, cv2.COLOR_RGB2GRAY)
        else:
            return np_array
    
    def _extract_landmarks_points(self, gray_image: np.ndarray) -> list:
        """Extrae los 68 puntos de landmarks de la cara"""
        # Crear un rectángulo que cubra toda la imagen (asumiendo que ya es una cara recortada)
        height, width = gray_image.shape
        face_rect = dlib.rectangle(0, 0, width, height)
        
        landmarks = self.landmarks_detector(gray_image, face_rect)
        landmarks_points = []
        for i in range(68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            landmarks_points.append((x, y))
        return landmarks_points
    
    def _create_face_mask(self, image_shape: tuple, landmarks_points: list) -> np.ndarray:
        """Crea una máscara facial usando los landmarks y las rutas definidas"""
        # Extraer coordenadas según las rutas
        routes_coordinates = []
        for i in range(len(self.FACE_MASK_ROUTES) - 1):
            source_point = self.FACE_MASK_ROUTES[i]
            target_point = self.FACE_MASK_ROUTES[i + 1]
            
            source_coordinate = landmarks_points[source_point]
            target_coordinate = landmarks_points[target_point]
            routes_coordinates.append(source_coordinate)
        
        # Agregar el último punto para cerrar la forma
        routes_coordinates = routes_coordinates + [routes_coordinates[0]]
        
        # Crear máscara
        mask = np.zeros((image_shape[0], image_shape[1]))
        mask = cv2.fillConvexPoly(mask, np.array(routes_coordinates), 1)
        mask = mask.astype(np.bool_)
        
        return mask
    
    def _apply_mask_to_image(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Aplica la máscara a la imagen"""
        output = np.zeros_like(image)
        output[mask] = image[mask]
        return output

    def normaliceFace(self, imagen: Image.Image) -> Image.Image:
        img_base = imagen.copy()
        
        # Convertir PIL a numpy array para dlib
        img_array = np.array(img_base)
        gray = self._gray(img_array)
        
        # Extraer landmarks directamente (sin detección facial)
        landmarks_points = self._extract_landmarks_points(gray)
        
        # Crear máscara facial
        mask = self._create_face_mask(img_array.shape, landmarks_points)
        
        # Aplicar máscara
        masked_image = self._apply_mask_to_image(img_array, mask)
        
        # Convertir de vuelta a PIL
        return Image.fromarray(masked_image.astype(np.uint8))