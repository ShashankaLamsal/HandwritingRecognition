import pandas as pd
import numpy as np
from mltu.configs import BaseModelConfigs
from mltu.dataProvider import DataProvider
from mltu.preprocessors import ImageReader
from mltu.transformers import LabelIndexer, LabelPadding, ImageResizer

# Load model configurations
configs = BaseModelConfigs.load("Models/handwriting_recognition/configs.yaml")

# Load dataset
train_data = pd.read_csv("train_data.csv", dtype={"text_label": str}).values.tolist()



class DebugLabelPadding(LabelPadding):
    def __call__(self, data, label):
        padded_label = super().__call__(data, label)

        print(f"Label Before Padding: {label}")
        print(f"Label Length Before Padding: {len(label)}")

        # Ensure the returned structure remains consistent
        if isinstance(padded_label, tuple):
            _, padded_label = padded_label  # Extract only the label part
        
        padded_label = np.array(padded_label)  # Convert safely

        print(f"Label After Padding: {padded_label}")
        print(f"Padded Label Length: {padded_label.shape}")  # Should be (256,)
        print("-" * 50)

        return data, padded_label  # Maintain expected (data, label) structure


# Initialize DataProvider with DebugLabelIndexer
train_data_provider = DataProvider(
    dataset=train_data,
    skip_validation=True,
    batch_size=1,  # Use batch_size=1 for debugging
    data_preprocessors=[ImageReader()],
    transformers=[
        ImageResizer(configs.width, configs.height, keep_aspect_ratio=True),
        LabelIndexer(configs.vocab),  # Debugging LabelIndexer
        DebugLabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab))
    ],
)

# Fetch and print first batch
for images, labels in train_data_provider:
    print(f"Final Label After Padding: {labels[0]}")
    print(f"Final Label Shape: {np.array(labels[0]).shape}")
    print("=" * 50)
    break
