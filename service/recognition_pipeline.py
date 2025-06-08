import logging
from typing import List, Optional, Tuple
import numpy as np
from PIL import Image
import io
from .detection.retina_face import RetinaFaceDetector
from .recognition.deepface_recognition import DeepFaceRecognition

logger = logging.getLogger(__name__)


class FaceRecognitionPipeline:
    def __init__(self):
        self.detector = RetinaFaceDetector()
        self.recognition = DeepFaceRecognition()

    def compare_embeddings(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        return self.recognition.compute_similarity(embedding1, embedding2)

    def process_image(self, image_input) -> Tuple[Optional[np.ndarray], float, Optional[str]]:
        """
        Process an image through the complete pipeline:
        1. Detect faces
        2. Generate embedding for the largest face
        """
        try:
            # Convertir input a PIL Image
            image = self._convert_to_pil_image(image_input)
            
            # Obtener caras recortadas y alineadas
            faces = self.detector.get_faces_choped_and_aligned(image)
            logger.info(f"Detected {len(faces)} faces in image")

            if not faces:
                return None, 0.0, "No faces detected in image"

            # Obtener la cara más grande
            largest_face = self._get_largest_face(faces)

            # Generar embedding
            embedding = self.recognition.generate_embedding(largest_face)
            confidence = 1.0

            return embedding, confidence, None

        except Exception as e:
            logger.error(f"Error in recognition pipeline: {str(e)}")
            return None, 0.0, f"Pipeline processing failed: {str(e)}"

    def process_images(self, images: List[Image.Image]) -> List[Tuple[Optional[np.ndarray], float, Optional[str]]]:
        """
        Process multiple images through the pipeline.
        """
        return [self.process_image(img) for img in images]

    def get_largest_face_embedding(self, image: Image.Image) -> Optional[np.ndarray]:
        """
        Process an image and return the embedding of the largest detected face.
        """
        try:
            faces = self.detector.get_faces_choped_and_aligned(image)
            
            if not faces:
                return None

            largest_face = self._get_largest_face(faces)
            return self.recognition.generate_embedding(largest_face)
            
        except Exception as e:
            logger.error(f"Error getting largest face embedding: {str(e)}")
            return None

    def process_all_faces(self, image_input) -> List[Tuple[Optional[np.ndarray], float, Optional[str]]]:
        """
        Process an image and generate embeddings for all detected faces.
        """
        try:
            # Convertir input a PIL Image
            image = self._convert_to_pil_image(image_input)
            
            # Obtener todas las caras recortadas y alineadas
            faces = self.detector.get_faces_choped_and_aligned(image)
            logger.info(f"Processing {len(faces)} faces")

            results = []
            for i, face in enumerate(faces):
                try:
                    embedding = self.recognition.generate_embedding(face)
                    results.append((embedding, 1.0, None))
                except Exception as e:
                    logger.error(f"Error processing face {i}: {str(e)}")
                    results.append((None, 0.0, f"Face processing failed: {str(e)}"))

            return results

        except Exception as e:
            logger.error(f"Error in process_all_faces: {str(e)}")
            return [(None, 0.0, f"Pipeline processing failed: {str(e)}")]

    def _convert_to_pil_image(self, image_input) -> Image.Image:
        """
        Convierte diferentes tipos de input a PIL Image
        """
        if isinstance(image_input, Image.Image):
            return image_input
        elif hasattr(image_input, 'read'):
            # Es un objeto de archivo
            image_input.seek(0)
            return Image.open(image_input).convert("RGB")
        elif isinstance(image_input, str):
            # Es una ruta de archivo
            return Image.open(image_input).convert("RGB")
        else:
            raise ValueError("Input type not supported")

    def _get_largest_face(self, faces: List[Image.Image]) -> Image.Image:
        """
        Encuentra la cara más grande por área (ancho * alto)
        """
        if not faces:
            raise ValueError("No faces provided")
        
        # Encontrar la cara con mayor área
        largest_face = max(faces, key=lambda face: face.size[0] * face.size[1])
        return largest_face