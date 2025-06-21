from typing import List, Tuple
from PIL import Image, ImageDraw
import io
from .retina_face import RetinaFaceDetector

from ..utils import convert_to_webp, drawFaces


class FaceDetection:
    def __init__(self):
        self.face_detection = RetinaFaceDetector()

    def detectAndDrawFaces(self, image_file):
        # Obtener coordenadas de las caras
        face_coords = self.give_face_cordenates(image_file)
        faces_found = len(face_coords)
        image_file.seek(0)
        if faces_found > 0:
            webp_bytes = drawFaces(image_file, face_coords)
        else:
            # Si no hay caras, igual convertimos la imagen a WebP
            image = Image.open(image_file)
            webp_bytes = convert_to_webp(image)
        return webp_bytes, "image/webp", faces_found

    def give_face_cordenates(self, image_file) -> List[Tuple[int, int, int, int]]:
        """
        Devuelve una lista de coordenadas (x1, y1, x2, y2) de las caras encontradas en la imagen.
        """
        # Simulación: retorna una lista con una cara ficticia
        return self.face_detection.detect_faces(image_file)
