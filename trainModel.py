import tensorflow as tf
from mltu.model_utils import residual_block
from keras import layers
from keras.models import Model
from mltu.losses import CTCloss
from mltu.metrics import CWERMetric
from mltu.configs import BaseModelConfigs
import pickle



# THIS ISNT THE MAIN TRAINING SCRIPT, THIS IS TO PREPARE TRAINING THE INTIAL MODEL 
# Load model configurations
configs = BaseModelConfigs.load("Models/handwriting_recognition/configs.yaml")

# Load dataset
with open("preprocessed_data.pkl", "rb") as f:
    dataset, vocab, max_len = pickle.load(f)

# Define Model Architecture
def build_model(input_dim, output_dim, activation='leaky_relu', dropout=0.2):
    inputs = layers.Input(shape=input_dim, name="input")

    # Normalize images
    x = layers.Lambda(lambda x: x / 255)(inputs)

    x = residual_block(x, 16, activation=activation, skip_conv=True, strides=1, dropout=dropout)
    x = residual_block(x, 16, activation=activation, skip_conv=True, strides=2, dropout=dropout)
    x = residual_block(x, 32, activation=activation, skip_conv=True, strides=2, dropout=dropout)
    x = residual_block(x, 64, activation=activation, skip_conv=True, strides=2, dropout=dropout)

    # Reshape for sequence processing
    x = layers.Reshape((x.shape[-3] * x.shape[-2], x.shape[-1]))(x)

    # BiLSTM for sequence learning
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(dropout)(x)

    outputs = layers.Dense(output_dim + 1, activation='softmax', name="output")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# Create model
model = build_model(
    input_dim=(configs.height, configs.width, 3),
    output_dim=len(configs.vocab)
)

# Compile Model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=configs.learning_rate),
    loss=CTCloss(),
    metrics=[CWERMetric(padding_token=len(configs.vocab))]
)

# Display Model Summary
model.summary(line_length=110)
