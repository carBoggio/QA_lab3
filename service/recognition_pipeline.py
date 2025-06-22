import logging
from typing import List, Optional, Tuple
import numpy as np
from PIL import Image
import cv2

# Componentes del Pipeline
from .detection.retina_face import RetinaFaceDetector
from .recognition.ArcFaceRecognition import ArcFaceDeepFaceRecognition
from .clasifier.knn_classifier import KNNFaceClassifier

# Clases Base
from .clasifier.Clasifier import BaseFaceClassifier
from .detection.detector import Detector
from .recognition.Recognition import Recognition
from .BaseFaceRecognitionPipeline import BaseFaceRecognitionPipeline
from .alignment.FaceAligner import FaceAligner
from .normalization.BasicNormalizer import BasicNormalizer
from .normalization.MediaPipeFaceNormalizer import MediaPipeFaceNormalizer
from .alignment.MediapipeFaceAligner import MediaPipeFaceAligner
from .clasifier.SVMFaceClassifier import SVMFaceClassifier
# Utilidades
from .utils import (
    convert_to_pil_image, drawFaces, crop_faces, resize_with_padding,
    show_step_visualization, show_processing_stages, show_processing_stages_combined,
    cleanup_visualization_windows
)

logger = logging.getLogger(__name__)


class FaceRecognitionPipeline(BaseFaceRecognitionPipeline):
    def __init__(self):
        self.detector: Detector = RetinaFaceDetector()
        self.normalizer: BasicNormalizer = MediaPipeFaceNormalizer()
        self.aligner: FaceAligner = MediaPipeFaceAligner()
        self.recognition: Recognition = ArcFaceDeepFaceRecognition()
        self.classifier: BaseFaceClassifier = SVMFaceClassifier()

    def extract_embedding_from_single_largest_face_image(
        self, face_image: Image.Image, show_visualization: bool = False
    ) -> Optional[np.ndarray]:
        """Orquesta el flujo completo de reconocimiento facial con visualizaciones."""
        
        # 1. Carga y Estandarización
        image_pil = convert_to_pil_image(face_image)
        
        if show_visualization:
            logger.info("Paso 1: Imagen Original")
            show_step_visualization(image_pil, "1_Imagen_Original", (0, 0))
        
        # 2. Detección de Caras
        face_coords = self.detector.detect_faces(image_pil)
        if not face_coords:
            logger.warning("No se detectaron caras en la imagen")
            return None
        
        # 3. Recorte de la cara más grande
        face_crop = image_pil.crop(face_coords[0])
        
        if show_visualization:
            logger.info("Paso 2: Cara Detectada y Recortada")
            show_step_visualization(face_crop, "2_Cara_Detectada", (100, 100))

        # 4. Alineación Geométrica usando el aligner (con fallback automático)
        try:
            aligned_face = self.aligner.align(face_crop)
        except Exception as e:
            logger.warning(f"Error en alineación: {e}, usando imagen de la cara recortada")
            aligned_face = face_crop  # Fallback a la cara recortada
        
        if show_visualization:
            logger.info("Paso 3: Cara Alineada")
            show_step_visualization(aligned_face, "3_Cara_Alineada", (200, 200))

        # 5. Normalización usando el normalizer (con fallback automático)
        try:    
            normalized_face = self.normalizer.normaliceFace(aligned_face)
        except Exception as e:
            logger.warning(f"Error en normalización: {e}, usando imagen alineada")
            normalized_face = aligned_face  # Fallback a la cara alineada
        
        if show_visualization:
            logger.info("Paso 4: Cara Normalizada")
            show_step_visualization(normalized_face, "4_Cara_Normalizada", (300, 300))

        # 6. Resize con padding
        img_resize = resize_with_padding(normalized_face)

        if show_visualization:
            logger.info("Paso 5: Resize")
            show_step_visualization(img_resize, "5_Resize", (400, 400))

        # 7. Generación de Embedding - Convertir a formato BGR para el modelo
        try:
            # Convertir la imagen PIL a array numpy en formato BGR
            face_array_bgr = cv2.cvtColor(np.array(img_resize), cv2.COLOR_RGB2BGR)
            embedding = self.recognition.generate_embedding(face_array_bgr)
            
            if show_visualization and embedding is not None:
                logger.info(f"✅ Paso 6: Embedding generado exitosamente - Dimensiones: {embedding.shape}")
                show_step_visualization(img_resize, "6_Embedding_Generado", (500, 500))
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generando embedding: {e}")
            return None

    def predict_people_identity_from_picture(
        self, picture: Image.Image, show_visualization: bool = True
    ) -> List[Tuple[str, str]]:
        """Predice identidades de todas las caras en la imagen."""
        
        image_pil = convert_to_pil_image(picture)
        
        if show_visualization:
            logger.info("Paso 1: Imagen Original")
            show_step_visualization(image_pil, "Imagen_Original", (0, 0))
        
        # Detectar caras
        face_coords = self.detector.detect_faces(image_pil)
        
        if not face_coords:
            logger.info("No se detectaron caras en la imagen")
            return []
        
        results = []
        
        for i, coord in enumerate(face_coords):
            logger.info(f"Procesando cara {i+1}/{len(face_coords)}")
            
            # Recortar cara
            face_crop = image_pil.crop(coord)
            
            if show_visualization:
                logger.info(f"Cara {i+1} - Paso 2: Cara Detectada y Recortada")
                show_step_visualization(face_crop, f"Cara_{i+1}_Detectada", (100, 100))
            
            # Alinear usando el aligner (con fallback automático)
            try:
                aligned_face = self.aligner.align(face_crop)
            except Exception as e:
                logger.warning(f"Error alineando cara {i+1}: {e}, usando imagen original")
                aligned_face = face_crop  # Fallback a la cara recortada
            
            if show_visualization:
                logger.info(f"Cara {i+1} - Paso 3: Cara Alineada")
                show_step_visualization(aligned_face, f"Cara_{i+1}_Alineada", (200, 200))

            # Normalizar usando el normalizer (con fallback automático)
            try:
                normalized_face = self.normalizer.normaliceFace(aligned_face)
            except Exception as e:
                logger.warning(f"Error normalizando cara {i+1}: {e}, usando imagen alineada")
                normalized_face = aligned_face  # Fallback a la cara alineada
            
            if show_visualization:
                logger.info(f"Cara {i+1} - Paso 4: Cara Normalizada")
                show_step_visualization(normalized_face, f"Cara_{i+1}_Normalizada", (300, 300))

            # Generar embedding
            try:
                # Resize con padding
                img_resize = resize_with_padding(normalized_face)
                
                # Convertir la imagen PIL a array numpy en formato BGR
                face_array_bgr = cv2.cvtColor(np.array(img_resize), cv2.COLOR_RGB2BGR)
                embedding = self.recognition.generate_embedding(face_array_bgr)
                
                if show_visualization and embedding is not None:
                    logger.info(f"Cara {i+1} - Paso 5: Embedding Generado")
                    show_step_visualization(img_resize, f"Cara_{i+1}_Embedding", (400, 400))
                    
            except Exception as e:
                logger.warning(f"Error generando embedding para cara {i+1}: {e}")
                embedding = None
            
            if embedding is None:
                results.append(("no_identificado", "Sin embedding"))
                continue
            
            # Clasificar
            try:
                prediction_result = self.classifier.predict(embedding)
                if prediction_result:
                    results.append(prediction_result)
                    logger.info(f"Cara {i+1} identificada como: {prediction_result[0]}")
                    
                    if show_visualization:
                        logger.info(f"Cara {i+1} - Paso 6: Clasificación Completada - {prediction_result[0]}")
                        show_step_visualization(normalized_face, f"Cara_{i+1}_Clasificada_{prediction_result[0]}", (500, 500))
                else:
                    results.append(("no_identificado", "No clasificado"))
                    if show_visualization:
                        logger.info(f"Cara {i+1} - Paso 6: No Clasificada")
                        show_step_visualization(normalized_face, f"Cara_{i+1}_No_Clasificada", (500, 500))
            except Exception as e:
                logger.warning(f"Error clasificando cara {i+1}: {e}")
                results.append(("no_identificado", "Error clasificación"))
                if show_visualization:
                    logger.info(f"Cara {i+1} - Paso 6: Error en Clasificación")
                    show_step_visualization(normalized_face, f"Cara_{i+1}_Error_Clasificacion", (500, 500))
        
        return results

    def draw_faces_in_picture(
        self, picture: Image.Image, show_visualization: bool = True
    ) -> Image.Image:
        """Dibuja las caras detectadas en la imagen."""
        
        picture = convert_to_pil_image(picture)
        
        # Detectar caras usando el detector
        face_coords = self.detector.detect_faces(picture)
        
        if not face_coords:
            logger.info("No se detectaron caras para dibujar")
            return picture
        
        # Dibujar caras usando la utilidad
        result_image = drawFaces(picture, face_coords)
        
        # Mostrar resultado en una sola ventana
        if show_visualization:
            try:
                img_array = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
                h, w = img_array.shape[:2]
                
                # Crear canvas con espacio para texto
                canvas = np.zeros((h + 50, w, 3), np.uint8)
                canvas[40:40+h, :] = img_array
                
                # Añadir texto con instrucciones
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(canvas, "Presiona cualquier tecla para continuar...", (10, 30), font, 0.6, (0, 255, 0), 2)
                
                cv2.namedWindow("Caras_Detectadas", cv2.WINDOW_NORMAL)
                cv2.imshow("Caras_Detectadas", canvas)
                cv2.waitKey(0)  # Esperar hasta que se presione una tecla
                cv2.destroyWindow("Caras_Detectadas")
            except Exception as e:
                logger.warning(f"Error mostrando caras detectadas: {e}")
        
        return result_image

    def process_with_full_pipeline(
        self, image: Image.Image, show_each_step: bool = True
    ) -> dict:
        """Ejecuta el pipeline completo y retorna información detallada."""
        
        result = {
            'original_image': image,
            'faces_detected': [],
            'embeddings': [],
            'predictions': [],
            'processed_faces': []
        }
        
        image_pil = convert_to_pil_image(image)
        
        if show_each_step:
            logger.info("Paso 1: Imagen Original")
            show_step_visualization(image_pil, "Pipeline_Imagen_Original", (0, 0))
        
        # Detección
        face_coords = self.detector.detect_faces(image_pil)
        result['faces_detected'] = face_coords
        
        if not face_coords:
            return result
        
        for i, coord in enumerate(face_coords):
            logger.info(f"Procesando cara {i+1}/{len(face_coords)}")
            
            face_data = {
                'coordinates': coord,
                'crop': None,
                'aligned': None,
                'normalized': None,
                'embedding': None,
                'prediction': None
            }
            
            # Procesar cada cara
            face_crop = image_pil.crop(coord)
            face_data['crop'] = face_crop
            
            if show_each_step:
                logger.info(f"Cara {i+1} - Paso 2: Cara Detectada y Recortada")
                show_step_visualization(face_crop, f"Pipeline_Cara_{i+1}_Detectada", (100, 100))
            
            # Alineación (con fallback automático)
            try:
                aligned_face = self.aligner.align(face_crop)
            except Exception as e:
                logger.warning(f"Error alineando cara {i+1}: {e}, usando imagen original")
                aligned_face = face_crop  # Fallback a la cara recortada
            
            face_data['aligned'] = aligned_face
            if show_each_step:
                logger.info(f"Cara {i+1} - Paso 3: Cara Alineada")
                show_step_visualization(aligned_face, f"Pipeline_Cara_{i+1}_Alineada", (200, 200))
            
            # Normalización (con fallback automático)
            try:
                normalized_face = self.normalizer.normaliceFace(aligned_face)
            except Exception as e:
                logger.warning(f"Error normalizando cara {i+1}: {e}, usando imagen alineada")
                normalized_face = aligned_face  # Fallback a la cara alineada
            
            face_data['normalized'] = normalized_face
            if show_each_step:
                logger.info(f"Cara {i+1} - Paso 4: Cara Normalizada")
                show_step_visualization(normalized_face, f"Pipeline_Cara_{i+1}_Normalizada", (300, 300))
            
            # Embedding
            try:
                # Resize con padding
                img_resize = resize_with_padding(normalized_face)
                
                # Convertir la imagen PIL a array numpy en formato BGR
                face_array_bgr = cv2.cvtColor(np.array(img_resize), cv2.COLOR_RGB2BGR)
                embedding = self.recognition.generate_embedding(face_array_bgr)
                face_data['embedding'] = embedding
                
                if show_each_step and embedding is not None:
                    logger.info(f"Cara {i+1} - Paso 5: Embedding Generado")
                    show_step_visualization(img_resize, f"Pipeline_Cara_{i+1}_Embedding", (400, 400))
            except Exception as e:
                logger.warning(f"Error generando embedding cara {i}: {e}")
                face_data['embedding'] = None
            
            # Predicción
            if face_data['embedding'] is not None:
                try:
                    prediction = self.classifier.predict(face_data['embedding'])
                    face_data['prediction'] = prediction
                    if show_each_step:
                        logger.info(f"Cara {i+1} - Paso 6: Clasificación Completada - {prediction[0] if prediction else 'No identificado'}")
                        show_step_visualization(face_data['normalized'], f"Pipeline_Cara_{i+1}_Clasificada", (500, 500))
                except Exception as e:
                    logger.warning(f"Error prediciendo cara {i}: {e}")
                    face_data['prediction'] = ("no_identificado", "Error predicción")
                    if show_each_step:
                        logger.info(f"Cara {i+1} - Paso 6: Error en Clasificación")
                        show_step_visualization(face_data['normalized'], f"Pipeline_Cara_{i+1}_Error_Clasificacion", (500, 500))
            
            result['processed_faces'].append(face_data)
        
        return result

    def cleanup_visualization_windows(self):
        """Cierra todas las ventanas de visualización."""
        cleanup_visualization_windows()