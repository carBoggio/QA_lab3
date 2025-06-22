import io
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import json
import hashlib
import os

logger = logging.getLogger()


def convert_to_webp(image: Image.Image) -> bytes:
    """
    Convierte una imagen Pillow a formato WebP con alta compresión y retorna los bytes.
    """
    output = io.BytesIO()
    image.save(output, format="WEBP", quality=80, method=6)
    return output.getvalue()


def load_from_database(media_path: str = "media/lab1/") -> Dict[str, List[np.ndarray]]:
    from service.BaseFaceRecognitionPipeline import BaseFaceRecognitionPipeline
    from service.recognition_pipeline import FaceRecognitionPipeline
    
    # Crear directorio data si no existe
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    cache_file = data_dir / "lab1_emb.json"
    
    # Verificar si hay cambios en las carpetas
    current_hash = _calculate_folder_hash(media_path)
    
    # Cargar caché existente si existe
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Verificar si el hash coincide (no hay cambios)
            if cache_data.get('folder_hash') == current_hash:
                logger.info("✅ Usando datos en caché - no hay cambios en las carpetas")
                return _load_embeddings_from_cache(cache_data)
            else:
                logger.info("🔄 Cambios detectados en las carpetas - recargando datos")
        except Exception as e:
            logger.warning(f"⚠️ Error cargando caché: {e}")
    
    # Si no hay caché válido, cargar datos desde las carpetas
    logger.info("🔄 Cargando datos desde las carpetas...")
    pipeline: BaseFaceRecognitionPipeline = FaceRecognitionPipeline()
    
    try:
        media_dir = Path(media_path)

        if not media_dir.exists():
            logger.error(f"❌ Directorio {media_path} no existe")
            return {}

        embeddings_dict = {}
        person_folders = [f for f in media_dir.iterdir() if f.is_dir()]
        
        # Estadísticas para el JSON
        stats = {
            'total_folders': len(person_folders),
            'folders_info': {},
            'total_images': 0
        }

        logger.info(f"📊 Encontradas {len(person_folders)} carpetas de personas")

        for person_folder in person_folders:
            person_name = person_folder.name
            person_embeddings = _load_person_embeddings(pipeline, person_folder)
            
            # Contar imágenes en la carpeta
            supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            image_files = [
                f for f in person_folder.iterdir()
                if f.is_file() and f.suffix.lower() in supported_extensions
            ]
            image_count = len(image_files)

            if person_embeddings:
                embeddings_dict[person_name] = person_embeddings
                stats['folders_info'][person_name] = {
                    'image_count': image_count,
                    'embeddings_count': len(person_embeddings)
                }
                stats['total_images'] += image_count
                logger.info(f"✅ Cargados {len(person_embeddings)} embeddings para {person_name} ({image_count} imágenes)")
            else:
                stats['folders_info'][person_name] = {
                    'image_count': image_count,
                    'embeddings_count': 0
                }
                stats['total_images'] += image_count
                logger.warning(f"⚠️ No se encontraron embeddings válidos para {person_name} ({image_count} imágenes)")

        logger.info(f"✅ Cargados embeddings para {len(embeddings_dict)} personas")
        
        # Guardar en caché
        cache_data = {
            'folder_hash': current_hash,
            'stats': stats,
            'embeddings': _convert_embeddings_to_cache(embeddings_dict)
        }
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        logger.info(f"💾 Datos guardados en caché: {cache_file}")
        return embeddings_dict

    except Exception as e:
        logger.error(f"❌ Error cargando embeddings: {e}")
        return {}


def _calculate_folder_hash(media_path: str) -> str:
    """
    Calcula un hash basado en el contenido de las carpetas para detectar cambios.
    """
    media_dir = Path(media_path)
    if not media_dir.exists():
        return ""
    
    hash_data = []
    
    for person_folder in sorted(media_dir.iterdir()):
        if person_folder.is_dir():
            folder_info = {
                'name': person_folder.name,
                'files': []
            }
            
            for file_path in sorted(person_folder.iterdir()):
                if file_path.is_file():
                    # Incluir nombre, tamaño y tiempo de modificación
                    stat = file_path.stat()
                    folder_info['files'].append({
                        'name': file_path.name,
                        'size': stat.st_size,
                        'mtime': stat.st_mtime
                    })
            
            hash_data.append(folder_info)
    
    # Crear hash del contenido
    hash_string = json.dumps(hash_data, sort_keys=True)
    return hashlib.md5(hash_string.encode()).hexdigest()


def _convert_embeddings_to_cache(embeddings_dict: Dict[str, List[np.ndarray]]) -> Dict[str, List[List[float]]]:
    """
    Convierte embeddings numpy a formato JSON serializable.
    """
    cache_embeddings = {}
    for person_name, embeddings_list in embeddings_dict.items():
        cache_embeddings[person_name] = [embedding.tolist() for embedding in embeddings_list]
    return cache_embeddings


def _load_embeddings_from_cache(cache_data: dict) -> Dict[str, List[np.ndarray]]:
    """
    Carga embeddings desde el formato de caché a numpy arrays.
    """
    embeddings_dict = {}
    for person_name, embeddings_list in cache_data['embeddings'].items():
        embeddings_dict[person_name] = [np.array(embedding) for embedding in embeddings_list]
    return embeddings_dict


def _load_person_embeddings(pipeline, person_folder: Path) -> List[np.ndarray]:
    """
    Carga todos los embeddings de una persona desde su carpeta.

    Args:
        pipeline: Instancia del pipeline de reconocimiento facial
        person_folder: Carpeta con las imágenes de la persona

    Returns:
        List[np.ndarray]: Lista de embeddings válidos
    """
    embeddings = []
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    try:
        image_files = [
            f for f in person_folder.iterdir()
            if f.is_file() and f.suffix.lower() in supported_extensions
        ]

        for image_file in image_files:
            try:
                # Cargar imagen
                image = Image.open(image_file)

                # Extraer embedding usando el pipeline
                embedding = pipeline.extract_embedding_from_single_largest_face_image(image)
                
                # Solo agregar embeddings válidos
                if embedding is not None and embedding.size > 0:
                    embeddings.append(embedding)
                else:
                    logger.warning(f"⚠️ No se pudo extraer embedding válido de {image_file}")

            except Exception as e:
                logger.warning(f"⚠️ Error procesando {image_file}: {e}")

    except Exception as e:
        logger.error(f"❌ Error accediendo a carpeta {person_folder}: {e}")

    return embeddings

def crop_faces(
    image_file: Image.Image,
    coords: List[Tuple[int, int, int, int]],
    target_size: Tuple[int, int] = (112, 112),
) -> List[Image.Image]:
    """
    Devuelve una lista de imágenes Pillow recortadas de las caras detectadas,
    todas redimensionadas al tamaño especificado manteniendo proporción con padding.
    target_size: tupla (width, height) para el tamaño final de cada cara
    """
    image = convert_to_pil_image(image_file)
    cropped = []
    for x1, y1, x2, y2 in coords:
        face = image.crop((x1, y1, x2, y2))
        cropped.append(face)
    return cropped


def drawFaces(image_file: Image.Image, face_coords: List[Tuple[int, int, int, int]]):
    """
    Dibuja recuadros en la imagen en las coordenadas dadas y retorna la imagen modificada en WebP.
    Usa Pillow para dibujar y convertir a WebP.
    """
    image = convert_to_pil_image(image_file)
    draw = ImageDraw.Draw(image)
    for x1, y1, x2, y2 in face_coords:
        draw.rectangle([x1, y1, x2, y2], outline="lightgreen", width=19)
    return convert_to_webp(image)


def resize_with_padding(
    image: Image.Image,
    target_size: Tuple[int, int] = (112, 112),
    pad_color: Tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """
    Redimensiona una imagen manteniendo la proporción y agrega padding para alcanzar el tamaño objetivo.

    Args:
        image: Imagen PIL a redimensionar
        target_size: Tupla (width, height) del tamaño objetivo
        pad_color: Color del padding en RGB (por defecto negro)

    Returns:
        Image.Image: Imagen redimensionada con padding
    """
    target_width, target_height = target_size
    original_width, original_height = image.size

    # Calcular la escala para mantener proporción
    scale = min(target_width / original_width, target_height / original_height)

    # Calcular nuevo tamaño manteniendo proporción
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # Redimensionar la imagen
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Crear nueva imagen con el tamaño objetivo y fondo del color especificado
    padded_image = Image.new("RGB", target_size, pad_color)

    # Calcular posición para centrar la imagen redimensionada
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2

    # Pegar la imagen redimensionada en el centro
    padded_image.paste(resized_image, (paste_x, paste_y))

    return padded_image


def normalize_l2(embedding: np.ndarray) -> np.ndarray:
    """
    Normalización L2 del embedding
    """
    epsilon = 1e-4
    norm = np.linalg.norm(embedding)

    return embedding / (norm + epsilon)


def convert_to_pil_image(image_input) -> Image.Image:
    """
    Convierte diferentes tipos de input a PIL Image, corrigiendo la orientación EXIF.
    """
    image = None
    if isinstance(image_input, Image.Image):
        image = image_input
    elif hasattr(image_input, "read"):
        # Es un objeto de archivo
        image_input.seek(0)
        image = Image.open(image_input)
    elif isinstance(image_input, str):
        # Es una ruta de archivo
        image = Image.open(image_input)
    else:
        raise ValueError("Input type not supported for image conversion.")

    # Corregir la orientación de la imagen usando los metadatos EXIF
    image = ImageOps.exif_transpose(image)

    # Asegurarse de que la imagen esté en formato RGB para consistencia
    return image.convert("RGB")


def show_step_visualization(image: Image.Image, step_name: str, window_pos: Tuple[int, int] = (0, 0)):
    """Muestra una imagen en una ventana con nombre específico."""
    try:
        img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        cv2.namedWindow(step_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(step_name, window_pos[0], window_pos[1])
        cv2.imshow(step_name, img_array)
        
        # Usar un timeout en lugar de esperar indefinidamente
        key = cv2.waitKey(3000)  # Esperar 3 segundos o hasta que se presione una tecla
        cv2.destroyWindow(step_name)
        
        # Si se presiona 'q' o 'ESC', salir
        if key in [ord('q'), ord('Q'), 27]:  # 27 es ESC
            logger.info("Visualización interrumpida por el usuario")
            
    except Exception as e:
        logger.warning(f"Error mostrando {step_name}: {e}")
        # Intentar cerrar la ventana si existe
        try:
            cv2.destroyWindow(step_name)
        except:
            pass


def show_processing_stages(original_image: Image.Image, face_crop: Image.Image, 
                          aligned_face: Image.Image, normalized_face: Image.Image,
                          stage_name: str = "Pipeline_Completo"):
    """Muestra cada etapa del pipeline una por una."""
    try:
        # Convertir imágenes PIL a arrays BGR para OpenCV
        original_np = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
        crop_np = cv2.cvtColor(np.array(face_crop), cv2.COLOR_RGB2BGR)
        aligned_np = cv2.cvtColor(np.array(aligned_face), cv2.COLOR_RGB2BGR)
        normalized_np = cv2.cvtColor(np.array(normalized_face), cv2.COLOR_RGB2BGR)
        
        # Mostrar imagen original
        show_step_visualization(original_image, f"{stage_name}_Original", (0, 0))
        
        # Mostrar cara recortada
        show_step_visualization(face_crop, f"{stage_name}_Detectada", (100, 100))
        
        # Mostrar cara alineada
        show_step_visualization(aligned_face, f"{stage_name}_Alineada", (200, 200))
        
        # Mostrar cara normalizada
        show_step_visualization(normalized_face, f"{stage_name}_Normalizada", (300, 300))

    except Exception as e:
        logger.warning(f"Error mostrando etapas de procesamiento: {e}")


def show_processing_stages_combined(original_image: Image.Image, face_crop: Image.Image, 
                                  aligned_face: Image.Image, normalized_face: Image.Image,
                                  stage_name: str = "Pipeline_Completo"):
    """Muestra todas las etapas del pipeline en una sola ventana (método alternativo)."""
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
        
        # Usar un timeout en lugar de esperar indefinidamente
        key = cv2.waitKey(3000)  # Esperar 3 segundos o hasta que se presione una tecla
        cv2.destroyWindow(stage_name)
        
        # Si se presiona 'q' o 'ESC', salir
        if key in [ord('q'), ord('Q'), 27]:  # 27 es ESC
            logger.info("Visualización interrumpida por el usuario")

    except Exception as e:
        logger.warning(f"Error mostrando etapas de procesamiento: {e}")


def cleanup_visualization_windows():
    """Cierra todas las ventanas de visualización."""
    try:
        cv2.destroyAllWindows()
    except Exception as e:
        logger.warning(f"Error cerrando ventanas: {e}")



