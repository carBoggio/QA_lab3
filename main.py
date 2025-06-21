from sklearn.neighbors import KNeighborsClassifier
from service.clasifier.knn_classifier import get_knn_classifier
from service.utils import load_from_database
from service.recognition_pipeline import FaceRecognitionPipeline
from PIL import Image
import cv2
import numpy as np
from service.normalization.DlibFaceNormalizer import DlibFaceNormalizer
from service.utils import resize_with_padding
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pipeline:
pipeline = FaceRecognitionPipeline()

# things for recognition
classificatory = get_knn_classifier()
classificatory.train()

img = Image.open("media/lab2/test.jpg")

# Prueba del pipeline completo
def test_pipeline():
    print("=== PRUEBA DEL PIPELINE DE RECONOCIMIENTO FACIAL ===")
    
    # 1. Detectar caras en la imagen
    print("\n1. Detectando caras...")
    faces = pipeline.detect_faces_in_image(img)
    print(f"   Se detectaron {len(faces)} caras")
    
    if not faces:
        print("   No se detectaron caras en la imagen")
        return
    
    # 2. Extraer embeddings de todas las caras
    print("\n2. Extrayendo embeddings...")
    embeddings_results = pipeline.detect_and_extract_all_face_embeddings(img)
    
    successful_embeddings = []
    for i, (embedding, confidence, error) in enumerate(embeddings_results):
        if embedding is not None:
            print(f"   Cara {i+1}: Embedding extraído exitosamente")
            successful_embeddings.append((embedding, confidence))
        else:
            print(f"   Cara {i+1}: Error - {error}")
    
    if not successful_embeddings:
        print("   No se pudieron extraer embeddings válidos")
        return
    
    # 3. Clasificar cada cara
    print("\n3. Clasificando caras...")
    for i, (embedding, confidence) in enumerate(successful_embeddings):
        try:
            identity, confidence_score = pipeline.classify_person_identity_from_embedding(embedding)
            print(f"   Cara {i+1}: {identity} (confianza: {confidence_score:.2f})")
        except Exception as e:
            print(f"   Cara {i+1}: Error en clasificación - {str(e)}")

# Prueba de normalización individual
def test_normalization():
    print("\n=== PRUEBA DE NORMALIZACIÓN ===")
    
    # Detectar una cara
    faces = pipeline.detect_faces_in_image(img)
    if not faces:
        print("No se detectaron caras para probar normalización")
        return
    
    face = faces[0]
    print(f"Imagen original: {face.size}")
    
    # Normalizar la cara
    try:
        normalizer = DlibFaceNormalizer()
        normalized_face = normalizer.normaliceFace(face)
        print(f"Imagen normalizada: {normalized_face.size}")
        print("Normalización exitosa")
    except Exception as e:
        print(f"Error en normalización: {str(e)}")

# Prueba de extracción de embedding con normalización
def test_embedding_with_normalization():
    print("\n=== PRUEBA DE EMBEDDING CON NORMALIZACIÓN ===")
    
    faces = pipeline.detect_faces_in_image(img)
    if not faces:
        print("No se detectaron caras para probar embedding")
        return
    
    face = faces[0]
    
    # Extraer embedding (esto incluye normalización automática)
    try:
        embedding = pipeline.extract_embedding_from_single_face_image(face)
        if embedding is not None:
            print(f"Embedding extraído exitosamente")
            print(f"Dimensiones del embedding: {embedding.shape}")
            print(f"Valores del embedding (primeros 5): {embedding[:5]}")
        else:
            print("No se pudo extraer el embedding")
    except Exception as e:
        print(f"Error extrayendo embedding: {str(e)}")

# Prueba de clasificación completa
def test_classification():
    print("\n=== PRUEBA DE CLASIFICACIÓN COMPLETA ===")
    
    try:
        # Extraer embedding de la imagen
        embedding = pipeline.extract_embedding_from_single_face_image(img)
        if embedding is not None:
            print(f"Embedding extraído: {embedding.shape}")
            
            # Clasificar
            identity, confidence = pipeline.classify_person_identity_from_embedding(embedding)
            print(f"Persona identificada: {identity}")
            print(f"Nivel de confianza: {confidence}")
        else:
            print("No se pudo extraer embedding")
    except Exception as e:
        print(f"Error en clasificación: {str(e)}")

if __name__ == "__main__":
    # Ejecutar todas las pruebas
    test_normalization()
    test_embedding_with_normalization()
    test_classification()
    test_pipeline()




   