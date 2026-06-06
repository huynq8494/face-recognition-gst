import onnxruntime as ort
import numpy as np
from torchvision import transforms
import sys
import os
from PIL import Image

sys.path.insert(0, "/home/huynq/study/repos/edgeface")
from face_alignment import align


def load_model(model_path: str):
    """Load ONNX model and return session."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    session = ort.InferenceSession(model_path)
    return session


def preprocess_image(image_path: str):
    """Preprocess image for inference."""

    # Load and resize image to 112x112
    image = Image.open(image_path).convert('RGB')
    image = image.resize((112, 112))

    # Transform to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    input_tensor = transform(image).unsqueeze(0)
    return input_tensor.numpy()


def run_inference(session, image_path: str):
    """Run inference on a single image."""
    # Preprocess
    input_data = preprocess_image(image_path)

    # Get input/output names
    input_name = session.get_inputs()[0].name

    # Run inference
    outputs = session.run(None, {input_name: input_data})

    return outputs[0]


# Example usage
if __name__ == "__main__":
    # Load model
    model_path = "/home/huynq/models/edgeface/edgeface_xs_gamma_06.onnx"
    session = load_model(model_path)

    # Run inference
    image_path = "/home/huynq/data/images/face_112x112.jpg"
    embedding = run_inference(session, image_path)

    print(f"Embedding shape: {embedding.shape}")
    np.set_printoptions(suppress=True)
    print(f"Embedding: {embedding}")
