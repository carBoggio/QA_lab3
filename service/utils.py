import io
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger()


def convert_to_webp(image: Image.Image) -> bytes:
    """
    Convierte una imagen Pillow a formato WebP con alta compresión y retorna los bytes.
    """
    output = io.BytesIO()
    image.save(output, format="WEBP", quality=80, method=6)
    return output.getvalue()


def load_from_database(media_path: str = "media/lab2/") -> Dict[str, List[np.ndarray]]:
    from service.BaseFaceRecognitionPipeline import BaseFaceRecognitionPipeline
    from service.recognition_pipeline import FaceRecognitionPipeline
    pipeline: BaseFaceRecognitionPipeline = FaceRecognitionPipeline()
    """
    Carga embeddings desde las carpetas de imágenes.

    Args:
        pipeline: Instancia del pipeline de reconocimiento facial
        media_path: Ruta base donde están las carpetas de personas

    Returns:
        Dict[str, List[np.ndarray]]: Diccionario con nombre de persona como clave
        y lista de embeddings como valor
    """
    try:
        media_dir = Path(media_path)

        if not media_dir.exists():
            logger.error(f"❌ Directorio {media_path} no existe")
            return {}

        embeddings_dict = {}
        person_folders = [f for f in media_dir.iterdir() if f.is_dir()]

        logger.info(f"📊 Encontradas {len(person_folders)} carpetas de personas")

        for person_folder in person_folders:
            person_name = person_folder.name
            person_embeddings = _load_person_embeddings(pipeline, person_folder)

            if person_embeddings:
                embeddings_dict[person_name] = person_embeddings
                logger.info(f"✅ Cargados {len(person_embeddings)} embeddings para {person_name}")
            else:
                logger.warning(f"⚠️ No se encontraron embeddings válidos para {person_name}")

        logger.info(f"✅ Cargados embeddings para {len(embeddings_dict)} personas")
        return embeddings_dict

    except Exception as e:
        logger.error(f"❌ Error cargando embeddings: {e}")
        return {}


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



