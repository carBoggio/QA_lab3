import os
from PIL import Image
from collections import defaultdict

from service.recognition_pipeline import FaceRecognitionPipeline
from service.clasifier.clasifier import get_knn_classifier

# Ruta base de las imágenes
BASE_DIR = "pictures"
USERS = ["lucas", "pedro"]
IMAGE_COUNT = 5  # número de imágenes por persona

# Inicializar pipeline y clasificador
pipeline = FaceRecognitionPipeline()
classifier = get_knn_classifier()

def cargar_embeddings():
    embeddings_dict = defaultdict(list)

    for user in USERS:
        user_dir = os.path.join(BASE_DIR, user)
        for i in range(1, IMAGE_COUNT + 1):
            image_path = os.path.join(user_dir, f"{user}{i}.jpg")
            try:
                with Image.open(image_path).convert("RGB") as img:
                    embedding, confidence, error = pipeline.process_image(img)
                    if embedding is not None:
                        embeddings_dict[user].append(embedding)
                    else:
                        print(f"[!] No se pudo procesar {image_path}: {error}")
            except Exception as e:
                print(f"[!] Error cargando imagen {image_path}: {e}")
    
    return embeddings_dict

def main():
    print("[+] Generando embeddings de entrenamiento...")
    embeddings = cargar_embeddings()
    print(embeddings)
    print("[+] Entrenando clasificador...")
    success = classifier.train_from_dict(embeddings)
    if not success:
        print("[!] Falló el entrenamiento del clasificador.")
        return

    print("[+] Clasificador entrenado con éxito.")

    # Probar reconocimiento con una imagen nueva
    test_image_path = "pictures/fotos_clase/pedro_lucas.jpg"
    try:
        with Image.open(test_image_path).convert("RGB") as img:
            embedding, _, error = pipeline.process_image(img)
            if embedding is not None:
                prediction = classifier.predict(embedding)
                print(prediction)
                if prediction:
                    print(f"[✓] Persona reconocida: {prediction}")
                else:
                    print("[✗] No se pudo reconocer la persona con suficiente confianza.")
            else:
                print(f"[!] Error procesando imagen: {error}")
    except Exception as e:
        print(f"[!] Error abriendo imagen de prueba: {e}")

if __name__ == "__main__":
    main()
