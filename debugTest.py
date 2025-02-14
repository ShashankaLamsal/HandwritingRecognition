import numpy as np
import tensorflow as tf
import cv2
from mltu.preprocessors import ImageReader
from mltu.transformers import LabelIndexer, ImageResizer
import onnxruntime as ort

# Load vocab
vocab = "!\"#&'(),-.0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWYZabcdefghijklmnopqrstuvwxyz"
vocab_dict = {i: c for i, c in enumerate(vocab)}

# Load model (change path if needed)
onxx_model_path = "Models/smallModel/model.onnx"
session = ort.InferenceSession(onxx_model_path)

# Preprocessing function
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Invalid image path or image not found.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = ImageResizer(128, 32, keep_aspect_ratio=True)(image, None)[0]  # Resize only image
    image = image.astype(np.float32) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Prediction function
def predict_text(image_path):
    image = preprocess_image(image_path)
    outputs = session.run(None, {session.get_inputs()[0].name: image})
    prediction = outputs[0]
    
    print("Raw Model Output:", prediction)
    print("Top Probabilities:", np.max(prediction, axis=-1))
    
    # Get predicted indices
    predicted_indices = np.argmax(prediction, axis=-1)[0]
    print("Predicted Indices:", predicted_indices)
    
    # Convert indices to characters
    predicted_text = "".join([vocab_dict[idx] for idx in predicted_indices if idx < len(vocab)])
    print("Decoded Text:", predicted_text)
    
    # Get confidence score (mean of top probabilities)
    confidence_score = np.mean(np.max(prediction, axis=-1)) * 100
    print(f"Confidence Score: {confidence_score:.2f}%")
    
    return predicted_text, confidence_score

# Test image (change to your test image path)
test_image = "uploads/test.png"
predict_text(test_image)
