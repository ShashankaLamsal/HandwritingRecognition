from mltu.configs import BaseModelConfigs

import pickle  # to laod vocab and max_len from a file

# Load preprocessed data
with open("preprocessed_data.pkl", "rb") as f:
    dataset, vocab, max_len = pickle.load(f)

configs = BaseModelConfigs()

# Saving vocabulary and max text length from preprocessing step
configs.vocab = "".join(sorted(vocab))  # Convert vocab set to a sorted string
configs.max_text_length = max_len  # Set max length from preprocessing

# Other configurations 
configs.height = 64  # Image height
configs.width = 256  # Image width
configs.batch_size = 32
configs.learning_rate = 0.001
configs.model_path = "Models/handwriting_recognition"

# Save configurations
configs.save()
print("Model configuration saved successfully!")
