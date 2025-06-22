from sklearn.neighbors import KNeighborsClassifier
from service.clasifier.SVMFaceClassifier import get_svm_classifier
from service.utils import load_from_database
from service.recognition_pipeline import FaceRecognitionPipeline
from PIL import Image
import cv2
import numpy as np
from service.normalization.DlibFaceNormalizer import DlibFaceNormalizer
from service.utils import resize_with_padding
import logging
import os

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Crear pipeline
pipeline = FaceRecognitionPipeline()

# Entrenar el clasificador
classifier = get_svm_classifier()
classifier.train()

def test_face_recognition():
    """Función para probar el reconocimiento facial con test1, test2 y test3"""
    
    # Rutas de las imágenes de test
    test_images = [
        "media/lab1/test1.jpg",
        "media/lab1/test2.jpg", 
        "media/lab1/test3.jpg"
    ]
    
    for i, test_image_path in enumerate(test_images, 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"PROBANDO TEST{i}")
        logger.info(f"{'='*50}")
        
        if not os.path.exists(test_image_path):
            logger.error(f"No se encontró la imagen: {test_image_path}")
            continue
        
        try:
            # Cargar la imagen
            logger.info(f"Cargando imagen: {test_image_path}")
            test_image = Image.open(test_image_path)
            
            # Probar reconocimiento con el método que reconoce gente
            logger.info("Iniciando reconocimiento facial...")
            results = pipeline.predict_people_identity_from_picture(
                test_image, 
                show_visualization=True
            )
            
            # Mostrar resultados
            logger.info(f"=== RESULTADOS DEL RECONOCIMIENTO TEST{i} ===")
            logger.info(f"Total de caras detectadas: {len(results)}")
            
            for j, (name, confidence) in enumerate(results):
                logger.info(f"Cara {j+1}: {name} (confianza: {confidence})")
            
            logger.info(f"=== FIN DE RESULTADOS TEST{i} ===")
            
        except Exception as e:
            logger.error(f"Error durante el test{i}: {e}")
    
    logger.info(f"\n{'='*50}")
    logger.info("TODOS LOS TESTS COMPLETADOS")
    logger.info(f"{'='*50}")

if __name__ == "__main__":
    test_face_recognition()


