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
# Utilidades
from .utils import convert_to_pil_image, drawFaces, crop_faces, resize_with_padding

logger = logging.getLogger(__name__)


class FaceRecognitionPipeline(BaseFaceRecognitionPipeline):
    def __init__(self):
        self.detector: Detector = RetinaFaceDetector()
        self.normalizer: BasicNormalizer = MediaPipeFaceNormalizer()
        self.aligner: FaceAligner = MediaPipeFaceAligner()
        self.recognition: Recognition = ArcFaceDeepFaceRecognition()
        self.classifier: BaseFaceClassifier = KNNFaceClassifier()

    def _show_step_visualization(self, image: Image.Image, step_name: str, window_pos: Tuple[int, int] = (0, 0)):
        """Muestra una imagen en una ventana con nombre específico."""
        try:
            img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            cv2.namedWindow(step_name, cv2.WINDOW_NORMAL)
            cv2.moveWindow(step_name, window_pos[0], window_pos[1])
            cv2.imshow(step_name, img_array)
            cv2.waitKey(500)  # Esperar 500ms para ver cada paso
        except Exception as e:
            logger.warning(f"Error mostrando {step_name}: {e}")

    def _show_processing_stages(self, original_image: Image.Image, face_crop: Image.Image, 
                               aligned_face: Image.Image, normalized_face: Image.Image,
                               stage_name: str = "Pipeline_Completo"):
        """Muestra todas las etapas del pipeline en una sola ventana."""
        try:
            # Convertir imágenes PIL a arrays BGR para OpenCV
            original_np = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
            crop_np = cv2.cvtColor(np.array(face_crop), cv2.COLOR_RGB2BGR)
            aligned_np = cv2.cvtColor(np.array(aligned_face), cv2.COLOR_RGB2BGR)
            normalized_np = cv2.cvtColor(np.array(normalized_face), cv2.COLOR_RGB2BGR)
            
            # Obtener dimensiones de las imágenes
            h1, w1 = original_np.shape[:2]
            h2, w2 = crop_np.shape[:2]
            h3, w3 = aligned_np.shape[:2]
            h4, w4 = normalized_np.shape[:2]
            
            # Encontrar la altura máxima para alinear todas las imágenes
            max_height = max(h1, h2, h3, h4)
            
            # Crear canvas con altura máxima y ancho total
            total_width = w1 + w2 + w3 + w4 + 40  # 40 para espacios entre imágenes
            vis = np.zeros((max_height + 60, total_width, 3), np.uint8)  # 60 para texto
            
            # Colocar imágenes en el canvas
            x_offset = 0
            
            # Imagen original
            vis[50:50+h1, x_offset:x_offset+w1] = original_np
            x_offset += w1 + 10
            
            # Cara recortada
            vis[50:50+h2, x_offset:x_offset+w2] = crop_np
            x_offset += w2 + 10
            
            # Cara alineada
            vis[50:50+h3, x_offset:x_offset+w3] = aligned_np
            x_offset += w3 + 10
            
            # Cara normalizada
            vis[50:50+h4, x_offset:x_offset+w4] = normalized_np
            
            # Añadir texto con etiquetas
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(vis, "Original", (10, 30), font, 0.7, (255, 255, 255), 2)
            cv2.putText(vis, "Detectada", (w1 + 20, 30), font, 0.7, (255, 255, 255), 2)
            cv2.putText(vis, "Alineada", (w1 + w2 + 30, 30), font, 0.7, (255, 255, 255), 2)
            cv2.putText(vis, "Normalizada", (w1 + w2 + w3 + 40, 30), font, 0.7, (255, 255, 255), 2)
            
            # Añadir instrucciones
            cv2.putText(vis, "Presiona cualquier tecla para continuar...", (10, max_height + 50), font, 0.6, (0, 255, 0), 2)
            
            # Mostrar la ventana
            cv2.namedWindow(stage_name, cv2.WINDOW_NORMAL)
            cv2.imshow(stage_name, vis)
            cv2.waitKey(0)  # Esperar hasta que se presione una tecla
            cv2.destroyWindow(stage_name)

        except Exception as e:
            logger.warning(f"Error mostrando etapas de procesamiento: {e}")

    def extract_embedding_from_single_largest_face_image(
        self, face_image: Image.Image, show_visualization: bool = True
    ) -> Optional[np.ndarray]:
        """Orquesta el flujo completo de reconocimiento facial con visualizaciones."""
        
        # 1. Carga y Estandarización
        image_pil = convert_to_pil_image(face_image)
        
        # 2. Detección de Caras
        face_coords = self.detector.detect_faces(image_pil)
        if not face_coords:
            logger.warning("No se detectaron caras en la imagen")
            return None
        
        # 3. Recorte de la cara más grande
        face_crop = image_pil.crop(face_coords[0])

        # 4. Alineación Geométrica usando el aligner
        try:
            aligned_face = self.aligner.align(face_crop)
        except Exception as e:
            logger.warning(f"Error en alineación: {e}")
            aligned_face = face_crop  # Fallback a la cara recortada

        # 5. Normalización usando el normalizer
        try:
            normalized_face = self.normalizer.normaliceFace(aligned_face)
        except Exception as e:
            logger.warning(f"Error en normalización: {e}")
            normalized_face = aligned_face  # Fallback a la cara alineada

        # Mostrar pipeline completo en una sola ventana
        if show_visualization:
            self._show_processing_stages(image_pil, face_crop, aligned_face, normalized_face)
        
        # 6. Generación de Embedding
        try:
            # Convertir la cara final a BGR para el modelo de reconocimiento
            face_array_bgr = cv2.cvtColor(np.array(normalized_face), cv2.COLOR_RGB2BGR)
            embedding = self.recognition.generate_embedding(face_array_bgr)
            
            if show_visualization and embedding is not None:
                logger.info(f"✅ Embedding generado exitosamente - Dimensiones: {embedding.shape}")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generando embedding: {e}")
            return None

    def predict_people_identity_from_picture(
        self, picture: Image.Image, show_visualization: bool = True
    ) -> List[Tuple[str, str]]:
        """Predice identidades de todas las caras en la imagen."""
        
        image_pil = convert_to_pil_image(picture)
        
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
            
            # Alinear usando el aligner
            try:
                aligned_face = self.aligner.align(face_crop)
            except Exception as e:
                logger.warning(f"Error alineando cara {i+1}: {e}")
                aligned_face = face_crop

            # Normalizar usando el normalizer
            try:
                normalized_face = self.normalizer.normaliceFace(aligned_face)
            except Exception as e:
                logger.warning(f"Error normalizando cara {i+1}: {e}")
                normalized_face = aligned_face

            # Mostrar pipeline para esta cara
            if show_visualization:
                self._show_processing_stages(image_pil, face_crop, aligned_face, normalized_face, f"Cara_{i+1}_Pipeline")

            # Generar embedding
            try:
                face_array_bgr = cv2.cvtColor(np.array(normalized_face), cv2.COLOR_RGB2BGR)
                embedding = self.recognition.generate_embedding(face_array_bgr)
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
                else:
                    results.append(("no_identificado", "No clasificado"))
            except Exception as e:
                logger.warning(f"Error clasificando cara {i+1}: {e}")
                results.append(("no_identificado", "Error clasificación"))
        
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
        
        # Detección
        face_coords = self.detector.detect_faces(image_pil)
        result['faces_detected'] = face_coords
        
        if not face_coords:
            return result
        
        for i, coord in enumerate(face_coords):
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
            
            # Alineación
            try:
                aligned_face = self.aligner.align(face_crop)
                face_data['aligned'] = aligned_face
            except Exception as e:
                logger.warning(f"Error alineando cara {i}: {e}")
                face_data['aligned'] = face_crop
            
            # Normalización
            try:
                normalized_face = self.normalizer.normaliceFace(face_data['aligned'])
                face_data['normalized'] = normalized_face
            except Exception as e:
                logger.warning(f"Error normalizando cara {i}: {e}")
                face_data['normalized'] = face_data['aligned']
            
            # Embedding
            try:
                face_array_bgr = cv2.cvtColor(np.array(face_data['normalized']), cv2.COLOR_RGB2BGR)
                embedding = self.recognition.generate_embedding(face_array_bgr)
                face_data['embedding'] = embedding
            except Exception as e:
                logger.warning(f"Error generando embedding cara {i}: {e}")
                face_data['embedding'] = None
            
            # Predicción
            if face_data['embedding'] is not None:
                try:
                    prediction = self.classifier.predict(face_data['embedding'])
                    face_data['prediction'] = prediction
                except Exception as e:
                    logger.warning(f"Error prediciendo cara {i}: {e}")
                    face_data['prediction'] = ("no_identificado", "Error predicción")
            
            result['processed_faces'].append(face_data)
            
            # Mostrar pipeline para esta cara
            if show_each_step:
                self._show_processing_stages(
                    image_pil, 
                    face_data['crop'], 
                    face_data['aligned'], 
                    face_data['normalized'],
                    f"Cara_{i+1}_Pipeline_Completo"
                )
        
        return result

    def cleanup_visualization_windows(self):
        """Cierra todas las ventanas de visualización."""
        try:
            cv2.destroyAllWindows()
        except Exception as e:
            logger.warning(f"Error cerrando ventanas: {e}")