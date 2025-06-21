import logging
from typing import List, Optional, Tuple
import numpy as np
from PIL import Image
from .detection.retina_face import RetinaFaceDetector
from .recognition.ArcFaceRecognition import ArcFaceDeepFaceRecognition
from .clasifier.knn_classifier import KNNFaceClassifier

from .clasifier.Clasifier import BaseFaceClassifier
from .detection.detector import Detector
from .recognition.Recognition import Recognition
from .BaseFaceRecognitionPipeline import BaseFaceRecognitionPipeline
from .utils import convert_to_pil_image, get_largest_face, drawFaces
from service.normalization.DlibFaceNormalizer import DlibFaceNormalizer
from service.normalization.BasicNormalizer import BasicNormalizer
logger = logging.getLogger(__name__)


class FaceRecognitionPipeline(BaseFaceRecognitionPipeline):
    def __init__(self):
        self.detector: Detector = RetinaFaceDetector()
        self.recognition: Recognition = ArcFaceDeepFaceRecognition()
        self.classifier: BaseFaceClassifier = KNNFaceClassifier()
        self.normalizer: BasicNormalizer = DlibFaceNormalizer()

    def extract_embedding_from_single_largest_face_image(
        self, face_image: Image.Image
    ) -> Optional[np.ndarray]:
        """
        Extrae embedding de la cara más grande en una imagen.
        Delega la detección al detector y la extracción de características al reconocedor.
        """
        # Convertir a PIL si es necesario
        face_image = convert_to_pil_image(face_image)
        
        # Detectar caras en la imagen
        face_coords = self.detector.detect_faces(face_image)
        
        if not face_coords:
            logger.warning("No se detectaron caras en la imagen")
            return None
        
        # Obtener la cara más grande
        largest_face = get_largest_face([face_image.crop(coord) for coord in face_coords])
        
        # Normalizar la cara
        normalized_face = self.normalizer.normalize(largest_face)
        
        # Extraer embedding
        embedding = self.recognition.extract_embedding(normalized_face)
        
        return embedding

    def predict_people_identity_from_picture(
        self, picture: Image.Image
    ) -> List[Tuple[str, str]]:
        """
        Predice la identidad de todas las personas en una foto.
        Delega la detección, extracción de características y clasificación a los componentes correspondientes.
        """
        # Convertir a PIL si es necesario
        picture = convert_to_pil_image(picture)
        
        # Detectar caras en la imagen
        face_coords = self.detector.detect_faces(picture)
        
        if not face_coords:
            logger.info("No se detectaron caras en la imagen")
            return []
        
        results = []
        
        for coord in face_coords:
            # Recortar cara
            face_crop = picture.crop(coord)
            
            # Normalizar cara
            normalized_face = self.normalizer.normalize(face_crop)
            
            # Extraer embedding
            embedding = self.recognition.extract_embedding(normalized_face)
            
            if embedding is None:
                results.append(("no_identificado", "No se pudo extraer características"))
                continue
            
            # Clasificar persona
            prediction_result = self.classifier.predict(embedding)
            
            if prediction_result:
                person_name, status = prediction_result
                results.append((person_name, status))
            else:
                results.append(("no_identificado", "No se pudo clasificar"))
        
        return results

    def draw_faces_in_picture(
        self, picture: Image.Image
    ) -> Image.Image:
        """
        Dibuja las caras detectadas en la imagen.
        Delega la detección al detector y el dibujo a la utilidad correspondiente.
        """
        # Convertir a PIL si es necesario
        picture = convert_to_pil_image(picture)
        
        # Detectar caras
        face_coords = self.detector.detect_faces(picture)
        
        if not face_coords:
            logger.info("No se detectaron caras para dibujar")
            return picture
        
        # Dibujar caras usando la utilidad
        return drawFaces(picture, face_coords)
   



