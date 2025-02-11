import tensorflow as tf
from mltu.configs import BaseModelConfigs
from mltu.dataProvider import DataProvider
#from mltu.callbacks import EarlyStopping,ModelCheckpoint, ReduceLROnPlateau, TensorBoard, TrainLogger, Model2onnx
from mltu.callbacks import TrainLogger, Model2onnx 
from mltu.preprocessors import ImageReader
from mltu.transformers import LabelIndexer, ImageResizer, LabelPadding
from mltu.losses import CTCloss

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

import pandas as pd  # Import pandas to read CSV

import pickle
import numpy as np


from itertools import islice

# Load model configurations
configs = BaseModelConfigs.load("Models/handwriting_recognition/configs.yaml")




# Load CSV manually
train_data = pd.read_csv("train_data.csv").values.tolist()
val_data = pd.read_csv("val_data.csv").values.tolist()

#train_data = pd.read_csv("train_data.csv", header=None, names=["image_path", "text_label"]).values.tolist()
#val_data = pd.read_csv("val_data.csv", header=None, names=["image_path", "text_label"]).values.tolist()






# Initialize DataProvider manually and load dataset
train_data_provider = DataProvider(
    dataset=train_data,
    skip_validation=True,
    batch_size=configs.batch_size,
    data_preprocessors=[ImageReader()],  # Required argument in MLtu 0.1.5
    transformers=[
        ImageResizer(configs.width, configs.height, keep_aspect_ratio=True),
        LabelIndexer(configs.vocab),
        LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab))
        ]
)

val_data_provider = DataProvider(
    dataset=val_data,
    skip_validation=True,
    batch_size=configs.batch_size,
    data_preprocessors=[ImageReader()],  # Required argument in MLtu 0.1.5
    transformers=[
        ImageResizer(configs.width, configs.height, keep_aspect_ratio=True),
        LabelIndexer(configs.vocab),
        LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab))
        ]
        
)


# Load compiled model
from trainModel import build_model
model = build_model(
    input_dim=(configs.height, configs.width, 3),
    output_dim=len(configs.vocab)
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=configs.learning_rate),
    loss=CTCloss(),
    metrics=["accuracy"]
)

# Define callbacks
earlystopper = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
checkpoint = ModelCheckpoint(f"{configs.model_path}/model.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
trainLogger = TrainLogger(configs.model_path)
tb_callback = TensorBoard(log_dir=f'{configs.model_path}/logs', update_freq=1)
reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=5, verbose=1, mode='auto')
model2onnx = Model2onnx(f"{configs.model_path}/model.h5")

#Debigging / batch checking and resizing




max_len = configs.max_text_length

label_indexer = LabelIndexer(configs.vocab)  # Converts text labels into numeric arrays

print(f"Total batches: {len(train_data_provider)}--------------------------------------------------------------")






# Train the model
model.fit(
    train_data_provider,
    validation_data=val_data_provider,
    epochs=configs.train_epochs,
    callbacks=[earlystopper, checkpoint, trainLogger, reduceLROnPlat, tb_callback, model2onnx],
    workers=configs.train_workers
)

# Save the training and validation datasets
train_data_provider.to_csv(f"{configs.model_path}/train.csv")
val_data_provider.to_csv(f"{configs.model_path}/val.csv")

print("Training complete. Model saved successfully!")
