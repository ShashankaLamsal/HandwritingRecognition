from mltu.dataProvider import DataProvider
from mltu.preprocessors import ImageReader
from mltu.transformers import ImageResizer, LabelIndexer, LabelPadding
from mltu.augmentors import RandomBrightness, RandomRotate, RandomErodeDilate
import pickle

# Load dataset, vocab, and max_len
with open("preprocessed_data.pkl", "rb") as f:
    dataset, vocab, max_len = pickle.load(f)

# Load model configurations
from mltu.configs import BaseModelConfigs
configs = BaseModelConfigs.load("Models/handwriting_recognition/configs.yaml")

# Define data preprocessing pipeline
data_provider = DataProvider(
    dataset=dataset,
    skip_validation=True,  # Skips dataset validation checks for speed
    batch_size=configs.batch_size,
    data_preprocessors=[ImageReader()],  # Reads images
    transformers=[
        ImageResizer(configs.width, configs.height, keep_aspect_ratio=False),  # Resizes images
        LabelIndexer(configs.vocab),  # Converts labels into numeric indexes
        LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab)),  # Pads labels
    ],
)

# Split dataset: 90% for training, 10% for validation
train_data_provider, val_data_provider = data_provider.split(split=0.9)

# Apply data augmentation (for training only)
train_data_provider.augmentors = [
    RandomBrightness(),  # Adjust brightness randomly
    RandomRotate(),  # Rotate images randomly
    RandomErodeDilate(),  # Apply erosion and dilation
]

# Save training and validation datasets for reproducibility
train_data_provider.to_csv("train_data.csv")
val_data_provider.to_csv("val_data.csv")

print("DataProvider setup completed! Training and validation datasets are ready.")
