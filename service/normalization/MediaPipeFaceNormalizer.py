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
        
        # Define face contour indices (MediaPipe's face mesh landmarks)
        # These indices define the outer boundary of the face
        self.FACE_OUTLINE_INDICES = [
            10,  338, 297, 332, 284, 251, 389, 356, 454, 323, 
            361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 
            176, 149, 150, 136, 172, 58,  132, 93,  234, 127, 
            162, 21,  54,  103, 67,  109
        ]
    
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
        
        # Get the face outline points
        face_outline = [points[i] for i in self.FACE_OUTLINE_INDICES]
        
        # Create a blank mask
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Draw convex hull of face outline
        hull = cv2.convexHull(np.array(face_outline))
        cv2.fillConvexPoly(mask, hull, 255)
        
        # Optional: Add some blur to smooth the mask edges
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        
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
        
        # Convert RGB to BGR if needed (MediaPipe expects RGB)
        if img_array.shape[2] == 3 and image.mode == 'RGB':
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Get face mask
        mask = self._get_face_mask(img_array)
        
        # Apply mask to each channel
        if len(img_array.shape) == 3:
            masked_img = np.zeros_like(img_array)
            for i in range(3):
                masked_img[:,:,i] = cv2.bitwise_and(img_array[:,:,i], mask)
        else:
            masked_img = cv2.bitwise_and(img_array, mask)
        
        # Convert back to PIL Image (RGB)
        if len(masked_img.shape) == 3:
            masked_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)
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