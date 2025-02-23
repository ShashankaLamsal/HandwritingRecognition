import tensorflow as tf
from mltu.configs import BaseModelConfigs
from mltu.preprocessors import ImageReader
from mltu.transformers import ImageResizer, LabelIndexer
import numpy as np
import cv2

# Load model configurations

configs = BaseModelConfigs.load("Models/smallModel/configs.yaml")

# Load the trained model

model = tf.keras.models.load_model("Models/smallModel/model.h5", compile=False)



def predict_text(image_path):
    image = cv2.imread(image_path)  # Load image directly
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB (if needed)
    image = cv2.resize(image, (configs.width, configs.height))  # Resize to match model input
    image = np.expand_dims(image, axis=0).astype(np.float32)  # Add batch dimension

    # Run model inference
    prediction = model.predict(image)

    # Get character predictions and their probabilities
    
    confidence_scores = np.max(prediction, axis=-1)[0]  # Get confidence for each character

    # Convert numerical predictions to text using the vocabulary
    predicted_indices = np.argmax(prediction, axis=-1)[0]  # Get the most probable character indices
    predicted_text = "".join([configs.vocab[idx] for idx in predicted_indices if idx < len(configs.vocab)])  # Convert indices to text

    # Compute overall confidence as the average confidence score
    avg_confidence = np.mean(confidence_scores) * 100  # Convert to percentage

    return predicted_text, avg_confidence


if __name__ == "__main__":
    #test_image = "uploads/h.png"  # Replace with your test image
    test_image = "Datasets/IAM_Words/words/a01/a01-000u/a01-000u-00-01.png" #word is move
    result, confidence = predict_text(test_image)
    print(f"Predicted Text: {result}")
    print(f"Confidence Score: {confidence:.2f}%")
