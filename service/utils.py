import io
from PIL import Image, ImageDraw
from typing import Tuple, List



def convert_to_webp(image: Image.Image) -> bytes:
    """
    Convierte una imagen Pillow a formato WebP con alta compresión y retorna los bytes.
    """
    output = io.BytesIO()
    image.save(output, format="WEBP", quality=80, method=6)
    return output.getvalue()



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
    
    image_file.seek(0)
    image = Image.open(image_file).convert("RGB")
    cropped = []
    for x1, y1, x2, y2 in coords:
        face = image.crop((x1, y1, x2, y2))
        
        face_padded = resize_with_padding(face, target_size)
        cropped.append(face_padded)
    return cropped


def drawFaces(image_file: Image.Image, face_coords: List[Tuple[int, int, int, int]]):
    """
    Dibuja recuadros en la imagen en las coordenadas dadas y retorna la imagen modificada en WebP.
    Usa Pillow para dibujar y convertir a WebP.
    """
    image = Image.open(image_file).convert("RGB")
    draw = ImageDraw.Draw(image)
    for x1, y1, x2, y2 in face_coords:
        draw.rectangle([x1, y1, x2, y2], outline="lightgreen", width=19)
    return convert_to_webp(image)


def resize_with_padding(
        image: Image.Image,
        target_size: Tuple[int, int],
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


