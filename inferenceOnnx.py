import onnxruntime as ort
import numpy as np
import cv2
from mltu.configs import BaseModelConfigs
from mltu.transformers import LabelIndexer

# Load model configurations
configs = BaseModelConfigs.load("Models/smallModel/configs.yaml")

# Load ONNX model
onnx_model_path = f"Models/smallModel/model.onnx"
session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])

# Define preprocessing function
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (configs.width, configs.height))  # Resize to match model input
    image = np.expand_dims(image, axis=0).astype(np.float32)  # Add batch dimension
    return image

# Define prediction function
def predict_text(image_path):
    image = preprocess_image(image_path)

    # Perform inference
    input_name = session.get_inputs()[0].name
    prediction = session.run(None, {input_name: image})[0]

    # Convert model output to text
    predicted_indices = np.argmax(prediction, axis=-1)[0]  # Get most probable character indices
    confidence_scores = np.max(prediction, axis=-1)[0]  # Get confidence for each character

    predicted_text = "".join([configs.vocab[idx] for idx in predicted_indices if idx < len(configs.vocab)])  
    avg_confidence = np.mean(confidence_scores) * 100  

    return predicted_text, avg_confidence
def predict_text1(image_path):
    image = preprocess_image(image_path)

    # Perform inference
    input_name = session.get_inputs()[0].name
    prediction = session.run(None, {input_name: image})[0]

    

    predicted_indices = np.argmax(prediction, axis=-1)[0]  
    confidence_scores = np.max(prediction, axis=-1)[0]  

    

    predicted_text = "".join([configs.vocab[idx] for idx in predicted_indices if idx < len(configs.vocab)])  
    avg_confidence = np.mean(confidence_scores) * 100  

    return predicted_text, avg_confidence

# Test the ONNX model
if __name__ == "__main__":
    test_image = "uploads/nerd.png"  #  test image path inside uploads
    result, confidence = predict_text1(test_image)
    print(f"Predicted Text: {result}")
    print(f"Confidence Score: {confidence:.2f}%")
