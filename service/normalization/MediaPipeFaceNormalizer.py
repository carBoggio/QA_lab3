import cv2
import numpy as np
import os
import requests
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import List, Union
from .BasicNormalizer import BasicNormalizer

class MediaPipeFaceNormalizer(BasicNormalizer):
    """
    Face normalizer implementation using MediaPipe facial landmarks with automatic model download.
    Masks everything outside the facial region with black.
    Handles frontal, profile, and 3/4 view faces.
    """
    
    # Shared model configuration with MediaPipeFaceAligner
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    MODEL_DIR = os.path.expanduser("~/.mediapipe/models")
    MODEL_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")
    
    def __init__(self):
        """
        Initialize the MediaPipe face normalizer.
        Automatically downloads the model if it doesn't exist locally.
        """
        super().__init__()
        
        # Ensure model is downloaded
        self._ensure_model_exists()
        
        # Initialize MediaPipe FaceLandmarker
        base_options = python.BaseOptions(model_asset_path=self.MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1)
        self.detector = vision.FaceLandmarker.create_from_options(options)
        
        # Define complete face contour indices (MediaPipe's face mesh landmarks)
        # Incluye todo el perímetro facial: frente, sienes, mejillas, mandíbula y barbilla
        self.FACE_OUTLINE_INDICES = [
            # Contorno facial completo en orden
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10
        ]
        
        # Landmarks adicionales para incluir más área facial
        self.FOREHEAD_INDICES = [9, 10, 151, 337, 299, 333, 298, 301]
        self.CHIN_INDICES = [175, 181, 84, 17, 314, 405, 320, 307]
        self.LEFT_CHEEK_INDICES = [116, 117, 118, 119, 120, 121, 126, 142]
        self.RIGHT_CHEEK_INDICES = [345, 346, 347, 348, 349, 350, 355, 371]
    
    def _ensure_model_exists(self):
        """Download the model if it doesn't exist locally."""
        if not os.path.exists(self.MODEL_PATH):
            print(f"Downloading MediaPipe face landmark model to {self.MODEL_PATH}...")
            os.makedirs(self.MODEL_DIR, exist_ok=True)
            
            try:
                response = requests.get(self.MODEL_URL, stream=True)
                response.raise_for_status()
                
                with open(self.MODEL_PATH, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print("Model downloaded successfully.")
            except Exception as e:
                raise RuntimeError(f"Failed to download model: {e}. Please download it manually from {self.MODEL_URL} and place it at {self.MODEL_PATH}")
    
    def _get_face_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Create a mask for the facial region.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Binary mask where face region is white (255) and background is black (0)
        """
        # Convert to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        
        # Detect landmarks
        detection_result = self.detector.detect(mp_image)
        
        if len(detection_result.face_landmarks) == 0:
            raise ValueError("No face detected in the image")
            
        # Get the first face's landmarks
        landmarks = detection_result.face_landmarks[0]
        
        # Convert landmarks to image coordinates
        h, w = image.shape[:2]
        points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        
        # Create a blank mask
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Método 1: Crear máscara más inclusiva usando múltiples regiones
        all_face_points = []
        
        # Agregar contorno principal
        face_outline = [points[i] for i in self.FACE_OUTLINE_INDICES if i < len(points)]
        all_face_points.extend(face_outline)
        
        # Agregar puntos de frente para incluir más área superior
        forehead_points = [points[i] for i in self.FOREHEAD_INDICES if i < len(points)]
        all_face_points.extend(forehead_points)
        
        # Agregar puntos de barbilla para incluir más área inferior
        chin_points = [points[i] for i in self.CHIN_INDICES if i < len(points)]
        all_face_points.extend(chin_points)
        
        # Agregar puntos de mejillas para incluir más área lateral
        cheek_points = [points[i] for i in self.LEFT_CHEEK_INDICES + self.RIGHT_CHEEK_INDICES if i < len(points)]
        all_face_points.extend(cheek_points)
        
        # Crear convex hull con todos los puntos
        if all_face_points:
            hull = cv2.convexHull(np.array(all_face_points))
            cv2.fillConvexPoly(mask, hull, 255)
        
        # Método 2: Expandir la máscara para incluir más área facial
        kernel = np.ones((15, 15), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Suavizar los bordes
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        
        # Asegurar que la máscara tenga valores binarios después del blur
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        return mask
    
    def normaliceFace(self, image: Image.Image) -> Image.Image:
        """
        Normalize a face in the image by masking non-facial regions.
        
        Args:
            image: PIL Image with a face
            
        Returns:
            PIL Image with only the facial region visible (rest masked in black)
        """
        # Convert to numpy array
        img_array = np.array(image)
        
        # Asegurar que tenemos RGB para MediaPipe
        if img_array.shape[2] == 3 and image.mode == 'RGB':
            # MediaPipe espera RGB, así que no convertimos a BGR
            rgb_array = img_array
        else:
            rgb_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        
        # Get face mask
        mask = self._get_face_mask(rgb_array)
        
        # Apply mask to each channel
        if len(img_array.shape) == 3:
            # Normalizar máscara a rango 0-1
            mask_normalized = mask.astype(np.float32) / 255.0
            
            # Aplicar máscara a cada canal
            masked_img = np.zeros_like(img_array)
            for i in range(3):
                masked_img[:,:,i] = (img_array[:,:,i].astype(np.float32) * mask_normalized).astype(np.uint8)
        else:
            masked_img = cv2.bitwise_and(img_array, mask)
        
        # Convert back to PIL Image
        return Image.fromarray(masked_img)
    
    def normaliceFaces(self, images: List[Image.Image]) -> List[Image.Image]:
        """
        Normalize multiple faces in a list of images.
        
        Args:
            images: List of PIL Images
            
        Returns:
            List of PIL Images with only facial regions visible
        """
        return super().normaliceFaces(images)