"""
Test script for Lost & Found AI API endpoints.
"""
import io
import os
import cv2
import numpy as np
from fastapi.testclient import TestClient
from app.main import app


def create_dummy_image():
    """Create a simple test image with a face-like shape."""
    # Create a 640x640 image (matching InsightFace det_size)
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    # Draw a simple face-like circle
    cv2.circle(img, (320, 320), 200, (200, 200, 200), -1)
    # Draw eyes
    cv2.circle(img, (260, 270), 30, (50, 50, 50), -1)
    cv2.circle(img, (380, 270), 30, (50, 50, 50), -1)
    
    # Encode as JPEG
    is_success, buffer = cv2.imencode(".jpg", img)
    if is_success:
        return io.BytesIO(buffer).getvalue()
    return None


# Use TestClient with the FastAPI app directly
client = TestClient(app)


def run_tests():
    print("Testing Endpoints...")
    print("=" * 50)

    # 1. Health check
    response = client.get("/api/health")
    print(f"GET /health: {response.status_code} - {response.text[:200]}")

    # 2. Text embedding
    response = client.post("/api/text-embedding", data={"text": "Lost red wallet with green ID card"})
    print(f"POST /text-embedding: {response.status_code} - {response.text[:200]}")

    # 3. Add text
    response = client.post("/api/add-text", data={"text": "Lost red wallet with green ID card", "post_id": "post_123"})
    print(f"POST /add-text: {response.status_code} - {response.text[:200]}")

    # 4. Search text
    response = client.post("/api/search-text", data={"text": "red wallet", "k": 5})
    print(f"POST /search-text: {response.status_code} - {response.text[:200]}")

    # Create dummy image
    img_bytes = create_dummy_image()
    if img_bytes:
        files = {"image": ("dummy.jpg", img_bytes, "image/jpeg")}

        # 5. Image embedding
        response = client.post("/api/image-embedding", files=files)
        print(f"POST /image-embedding: {response.status_code} - {response.text[:200]}")

        # 6. Add image
        response = client.post("/api/add-image", data={"post_id": "post_124"}, files=files)
        print(f"POST /add-image: {response.status_code} - {response.text[:200]}")

        # 7. Search image
        response = client.post("/api/search-image", data={"k": 5}, files=files)
        print(f"POST /search-image: {response.status_code} - {response.text[:200]}")

        # Face endpoints
        # 8. Add face
        response = client.post("/api/add-face", data={"person_id": "person_abc"}, files=files)
        print(f"POST /add-face: {response.status_code} - {response.text[:200]}")

        # 9. Face match
        response = client.post("/api/face-match", data={"k": 5}, files=files)
        print(f"POST /face-match: {response.status_code} - {response.text[:200]}")

        # 10. Multimodal search
        response = client.post("/api/multimodal-search", data={"text": "wallet", "k": 5}, files=files)
        print(f"POST /multimodal-search: {response.status_code} - {response.text[:200]}")
    else:
        print("Failed to create dummy image. Skipping image endpoints.")

    print("=" * 50)
    print("Tests completed.")


if __name__ == "__main__":
    run_tests()