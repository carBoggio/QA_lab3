from PIL import Image
import cv2
import numpy as np
from service.detection.retina_face import RetinaFaceDetector
from service.recognition.ArcFaceRecognition import ArcFaceDeepFaceRecognition
import os

def test_recognition_system():
    """Prueba el sistema completo de reconocimiento"""
    print("=== Probando Sistema de Reconocimiento ===")
    
    detector = RetinaFaceDetector()
    recognition = ArcFaceDeepFaceRecognition()
    
    # 1. Procesar fotos individuales de Pedro
    pedro_embeddings = []
    print("\n--- Procesando fotos de Pedro ---")
    for i in range(1, 6):
        photo_path = f"pictures/pedro/pedro{i}.jpg"
        if os.path.exists(photo_path):
            image = Image.open(photo_path)
            faces = detector.get_faces_choped_and_aligned(image)
            
            for face in faces:
                embedding = recognition.generate_embedding(face)
                if embedding.size > 0:
                    pedro_embeddings.append(embedding)
                    
            print(f"pedro{i}.jpg: {len(faces)} caras encontradas")
    
    # 2. Procesar fotos individuales de Lucas
    lucas_embeddings = []
    print("\n--- Procesando fotos de Lucas ---")
    for i in range(1, 6):
        photo_path = f"pictures/lucas/lucas{i}.jpg"
        if os.path.exists(photo_path):
            image = Image.open(photo_path)
            faces = detector.get_faces_choped_and_aligned(image)
            
            for face in faces:
                embedding = recognition.generate_embedding(face)
                if embedding.size > 0:
                    lucas_embeddings.append(embedding)
                    
            print(f"lucas{i}.jpg: {len(faces)} caras encontradas")
    
    print(f"\nEmbeddings de Pedro: {len(pedro_embeddings)}")
    print(f"Embeddings de Lucas: {len(lucas_embeddings)}")
    
    # 3. Procesar foto de clase
    print("\n--- Procesando foto de clase ---")
    class_photo = "pictures/fotos_clase/pedro_lucas.jpg"
    
    if os.path.exists(class_photo):
        image = Image.open(class_photo)
        class_faces = detector.get_faces_choped_and_aligned(image)
        
        class_embeddings = []
        for face in class_faces:
            embedding = recognition.generate_embedding(face)
            if embedding.size > 0:
                class_embeddings.append(embedding)
        
        print(f"Caras en foto de clase: {len(class_faces)}")
        print(f"Embeddings extraídos: {len(class_embeddings)}")
        
        # 4. Comparar cada cara de la clase con Pedro y Lucas
        print("\n--- Resultados de Reconocimiento ---")
        
        for i, class_embedding in enumerate(class_embeddings):
            print(f"\nCara {i+1}:")
            
            # Comparar con Pedro
            pedro_similarities = []
            for pedro_emb in pedro_embeddings:
                similarity = recognition.compute_similarity(class_embedding, pedro_emb)
                pedro_similarities.append(similarity)
            
            # Comparar con Lucas  
            lucas_similarities = []
            for lucas_emb in lucas_embeddings:
                similarity = recognition.compute_similarity(class_embedding, lucas_emb)
                lucas_similarities.append(similarity)
            
            # Mejores matches
            best_pedro = max(pedro_similarities) if pedro_similarities else 0
            best_lucas = max(lucas_similarities) if lucas_similarities else 0
            avg_pedro = np.mean(pedro_similarities) if pedro_similarities else 0
            avg_lucas = np.mean(lucas_similarities) if lucas_similarities else 0
            
            print(f"  Pedro - Mejor: {best_pedro:.3f}, Promedio: {avg_pedro:.3f}")
            print(f"  Lucas - Mejor: {best_lucas:.3f}, Promedio: {avg_lucas:.3f}")
            
            # Determinar identificación
            if best_pedro > best_lucas and best_pedro > 0.6:
                print(f"  -> Identificado como: PEDRO (confianza: {best_pedro:.3f})")
            elif best_lucas > best_pedro and best_lucas > 0.6:
                print(f"  -> Identificado como: LUCAS (confianza: {best_lucas:.3f})")
            elif max(best_pedro, best_lucas) > 0.4:
                name = "PEDRO" if best_pedro > best_lucas else "LUCAS"
                conf = max(best_pedro, best_lucas)
                print(f"  -> Posiblemente: {name} (confianza: {conf:.3f})")
            else:
                print(f"  -> NO IDENTIFICADO")
        
        # 5. Mostrar las caras detectadas
        print("\n--- Mostrando caras detectadas ---")
        for i, face in enumerate(class_faces):
            face_array = np.array(face)
            face_bgr = cv2.cvtColor(face_array, cv2.COLOR_RGB2BGR)
            cv2.imshow(f'Cara {i+1} de la clase', face_bgr)
        
        print("Presiona cualquier tecla para cerrar las ventanas...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    else:
        print(f"No se encontró la foto de clase: {class_photo}")

if __name__ == "__main__":
    try:
        test_recognition_system()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()