"""
Face recognition using InsightFace (ArcFace).
"""
import numpy as np
import cv2
from insightface.app import FaceAnalysis
from typing import Optional
import io
from PIL import Image


class FaceRecognizer:
    """Face embedding extraction using InsightFace with ArcFace model."""
    
    def __init__(self):
        # Initialize InsightFace with the Buffalo_L model (ArcFace backbone)
        self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
    
    def get_face_embedding(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """
        Extract face embedding from image bytes.
        
        Returns:
            512-dim normalized embedding vector, or None if no face is detected.
        """
        try:
            # Convert image bytes to numpy array (handle both RGB and grayscale)
            pil_img = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if necessary (handles grayscale, RGBA, etc.)
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            
            img = np.array(pil_img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Detect faces
            faces = self.app.get(img)
            if len(faces) == 0:
                return None
            
            # Use the first detected face (typically the largest/most prominent)
            face = faces[0]
            embedding = face.embedding  # numpy array of shape (512,)
            
            # Normalize the embedding to unit length
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            raise RuntimeError(f"Failed to extract face embedding: {str(e)}")
    
    @property
    def is_ready(self) -> bool:
        """Check if the face recognizer is properly initialized."""
        return self.app is not None