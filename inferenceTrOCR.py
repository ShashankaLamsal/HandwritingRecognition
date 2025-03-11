import torch
import torch.nn.functional as F
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

# Load the processor and model
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

def predict_text(image_path):
    
    if isinstance(image_path, str):  # If it's a file path, open image
        image = Image.open(image_path).convert("RGB")
    elif isinstance(image_path, Image.Image):  # If it's a PIL Image, use it directly
        image = image_path
    else:
        raise ValueError("Invalid input: Must be a file path or PIL Image.")
    # preprocess
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    # Run inference
    with torch.no_grad():
        outputs = model.generate(pixel_values, output_scores=True, return_dict_in_generate=True)

    # prediction
    predicted_text = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
    
    # confidence scores
    scores = torch.cat(outputs.scores, dim=0)  # Get logit scores for each predicted token
    probabilities = F.softmax(scores, dim=-1).max(dim=-1).values  # Get max probability per token
    avg_confidence = torch.mean(probabilities).item() * 100  # Average confidence score


    return predicted_text, avg_confidence
