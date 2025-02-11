import pandas as pd
from mltu.configs import BaseModelConfigs
from mltu.dataProvider import DataProvider
from mltu.preprocessors import ImageReader
from mltu.transformers import LabelIndexer, LabelPadding

# Load model configurations
configs = BaseModelConfigs.load("Models/handwriting_recognition/configs.yaml")

# Load dataset
train_data = pd.read_csv("train_data.csv").values.tolist()

# Initialize DataProvider
train_data_provider = DataProvider(
    dataset=train_data,
    skip_validation=True,
    batch_size=1,  # Use batch_size=1 for debugging
    data_preprocessors=[ImageReader()],
    transformers=[
        LabelIndexer(configs.vocab),  # Convert text labels to numeric first
        LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab))  # Then pad
    ],
)

# Fetch and print first batch
try:
    batch = next(iter(train_data_provider))
    images, labels = batch

    print(f"Raw Label (Before Indexing): {train_data[0][1]}")
    print(f"Indexed Label (After LabelIndexer): {labels[0]}")
    print(f"Indexed Label Shape: {labels[0].shape}")
    print(f"Max Text Length: {configs.max_text_length}")
    print(f"Vocab: {configs.vocab}")
    print(f"Vocab Length: {len(configs.vocab)}")
    print(f"Character Indices Mapping: { {char: i for i, char in enumerate(configs.vocab)} }")

except Exception as e:
    print(f"Error fetching batch: {e}")
