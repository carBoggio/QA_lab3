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
import os

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Crear pipeline
pipeline = FaceRecognitionPipeline()

# Cargar embeddings desde la base de datos
embeddings_dict = load_from_database()

# Procesar todas las imágenes de Larisa
larisa_folder = "media/lab2/Larisa"
larisa_images = [
    "Larisa.jpg",
    "Larisa2.jpg", 
    "Larisa3.jpg",
    "Larisa4.jpg",
    "Larisa5.jpg"
]

print("🔍 Procesando imágenes de Larisa...")
print("=" * 50)

for i, image_name in enumerate(larisa_images, 1):
    
    image_path = os.path.join(larisa_folder, image_name)

    if os.path.exists(image_path):
        print(f"\n📸 Procesando imagen {i}/5: {image_name}")
        print(f"📁 Ruta: {image_path}")
        
        try:
            # Cargar imagen
            img = Image.open(image_path)
            print(f"📐 Tamaño original: {img.size}")
            
            # Extraer embedding
            embedding = pipeline.extract_embedding_from_single_largest_face_image(img)
            
            if embedding is not None:
                print(f"✅ Embedding extraído exitosamente - Tamaño: {embedding.shape}")
            else:
                print("❌ No se pudo extraer embedding")
                
        except Exception as e:
            print(f"❌ Error procesando {image_name}: {e}")
    else:
        print(f"⚠️ Imagen no encontrada: {image_path}")

print("\n" + "=" * 50)
print("🎉 Procesamiento completado!")
print("💡 Presiona cualquier tecla para cerrar las ventanas de OpenCV...")
input()