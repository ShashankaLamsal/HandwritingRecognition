from mltu.configs import BaseModelConfigs

import pickle  # to laod vocab and max_len from a file

# Load preprocessed data
with open("preprocessed_data.pkl", "rb") as f:
    dataset, vocab, max_len = pickle.load(f)

configs = BaseModelConfigs()

# Saving vocabulary and max text length from preprocessing step
configs.vocab = "".join(sorted(vocab))  # Convert vocab set to a sorted string
configs.max_text_length = 32       # Set max length from preprocessing, 32, 256

# Other configurations                  2 values, 1 for testing: 1 for training
configs.height = 64                     # Image height  
configs.width = 256                     # Image width
configs.batch_size = 4                  #4 , 16
configs.learning_rate = 0.002              # faster learning rate but not efficient, 0.002, 0.0005
configs.model_path = "Models/handwriting_recognition"
configs.train_epochs = 2               # training epochs, 2, 50
configs.train_workers = 2                  #2 for test, 4 for efficient data loading

# Save configurations
configs.save()
print("Model configuration saved successfully!")

