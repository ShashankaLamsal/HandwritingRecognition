from inferenceTrOCR import predict_text

test_image = "uploads/conduction.png"

text, confidence = predict_text(test_image)
print(f"Predicted Text: {text}")
print(f"Confidence Score: {confidence:.2f}%")
