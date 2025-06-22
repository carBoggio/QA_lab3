from typing import List, Tuple
from PIL import Image
import numpy as np
from deepface import DeepFace
from .detector import Detector


class RetinaFaceDetector(Detector):
    def detect_faces(self, image_file: Image.Image) -> List[Tuple[int, int, int, int]]:
        img_array = self._parse_img(image_file)

        detections = DeepFace.analyze(
            img_path=img_array,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="retinaface",
        )
        return self._get_cords(detections)

    def get_faces_choped_and_aligned(
        self, image_file: Image.Image
    ) -> List[Image.Image]:
        img_array = self._parse_img(image_file)

        faces = DeepFace.extract_faces(
            img_path=img_array,
            enforce_detection=True,
            align=True,
            detector_backend="retinaface",
        )

        pil_faces = []
        for face in faces:
            face_img = face["face"]
            # Convertir a formato correcto
            if face_img.max() <= 1.0:
                face_img = (face_img * 255).astype(np.uint8)

            # Convertir a PIL Image
            pil_face = Image.fromarray(face_img)
            pil_faces.append(pil_face)

        return pil_faces

    def _parse_img(self, image_file) -> np.ndarray:
        if isinstance(image_file, str):
            # Si es una ruta de archivo
            image = Image.open(image_file).convert("RGB")
        elif hasattr(image_file, "read"):
            # Si es un objeto de archivo (BufferedReader, BytesIO, etc.)
            image = Image.open(image_file).convert("RGB")
        else:
            # Si ya es una imagen PIL
            image = image_file.convert("RGB")
        return np.array(image)

    def _get_cords(self, detections) -> List[Tuple[int, int, int, int]]:
        coords = []

        if isinstance(detections, list):
            for detection in detections:
                region = detection["region"]
                x, y, w, h = region["x"], region["y"], region["w"], region["h"]
                coords.append((x, y, x + w, y + h))
        else:
            region = detections["region"]
            x, y, w, h = region["x"], region["y"], region["w"], region["h"]
            coords.append((x, y, x + w, y + h))

        return coords
