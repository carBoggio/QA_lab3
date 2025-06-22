from typing import Tuple, Optional, Union, List
from abc import ABC, abstractmethod
from PIL import Image
import numpy as np
import cv2
import os
import requests
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from .FaceAligner import FaceAligner

class MediaPipeFaceAligner(FaceAligner):
    """
    Face aligner implementation using MediaPipe facial landmarks with automatic model download.
    Aligns faces based on eye positions to achieve consistent rotation and scale.
    """
    
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    MODEL_DIR = os.path.expanduser("~/.mediapipe/models")
    MODEL_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")
    
    def __init__(self, target_size: Tuple[int, int] = (112, 112)):
        """
        Initialize the MediaPipe face aligner.
        Automatically downloads the model if it doesn't exist locally.
        
        Args:
            target_size: Target size (width, height) for aligned faces
        """
        super().__init__(target_size)
        
        # Ensure model is downloaded
        self._ensure_model_exists()
        
        # Initialize MediaPipe FaceLandmarker
        base_options = python.BaseOptions(model_asset_path=self.MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
            num_faces=1)
        self.detector = vision.FaceLandmarker.create_from_options(options)
    
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
    
    def _get_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect facial landmarks using MediaPipe.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Array of facial landmarks or None if no face detected
        """
        # Convert to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        
        # Detect landmarks
        detection_result = self.detector.detect(mp_image)
        
        if len(detection_result.face_landmarks) == 0:
            return None
            
        # Get the first face's landmarks
        landmarks = detection_result.face_landmarks[0]
        return np.array([(lm.x * image.shape[1], lm.y * image.shape[0]) for lm in landmarks])

    def align(self, image: Union[Image.Image, np.ndarray]) -> Image.Image:
        # Convert to numpy array if it's a PIL Image
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert BGR to RGB if needed
        if image.shape[2] == 3 and image.dtype == 'uint8':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get facial landmarks
        landmarks = self._get_landmarks(image)
        if landmarks is None:
            raise ValueError("No face detected in the image")
        
        # Get eye centers
        left_eye = landmarks[468]
        right_eye = landmarks[473]
        
        # Calculate angle between the eyes
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))
        
        # Get center between eyes
        eyes_center = ((left_eye[0] + right_eye[0]) // 2,
                    (left_eye[1] + right_eye[1]) // 2)
        
        # Get rotation matrix WITHOUT scaling (scale = 1.0)
        M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)  # ✅ Solo rotación
        
        # Use original image size to avoid cropping
        h, w = image.shape[:2]
        
        # Apply affine transformation with original size
        aligned_face = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
        
        # Convert back to PIL Image
        return Image.fromarray(aligned_face)
      