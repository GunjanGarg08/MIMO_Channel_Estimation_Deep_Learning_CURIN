import os
import numpy as np
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras.callbacks import ModelCheckpoint

def add_label_noise(y, noise_level=0.1):
    noise = noise_level * np.std(y) * np.random.randn(*y.shape)
    return y + noise

def linear_model(input_shape):
    model = models.Sequential([
        layers.Flatten(input_shape=input_shape),
        layers.Dense(256, activation='linear', kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(0.5),
        layers.Dense(np.prod(input_shape), activation='linear'),
        layers.Reshape(input_shape)
    ])
    model.compile(optimizer='SGD', loss='mse', metrics=['mae'])
    return model

def nonlinear_model(input_shape):
    model = models.Sequential([
        layers.Reshape((input_shape[0], input_shape[1], 1), input_shape=input_shape),
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(np.prod(input_shape), activation='linear'),
        layers.Reshape(input_shape)
    ])
    model.compile(optimizer='SGD', loss='mse', metrics=['mae'])
    return model

def train_model(model, X_train, y_train, X_val, y_val, model_name, save_folder):
    os.makedirs(save_folder, exist_ok=True)
    model_path = os.path.join(save_folder, f'{model_name}.keras')

    checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', mode='min')
    model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_val, y_val), callbacks=[checkpoint])
    print(f"Model saved to {model_path}")

snr_values = [-10, -5, 0, 5, 10, 15, 20]
save_folder = "/workspaces/MIMO_Channel_Estimation_Deep_Learning_CURIN/Iterations/Trained_Model"

for snr in snr_values:
    dataset_path = f"/workspaces/MIMO_Channel_Estimation_Deep_Learning_CURIN/Iterations/Generated_Data/thz_mimo_dataset_snr_{snr}.npz"
    data = np.load(dataset_path)

    X_train, X_val = data['X_train'], data['X_val']
    y_train, y_val = data['y_train'], data['y_val']

    lin_model = linear_model(input_shape=X_train.shape[1:])
    train_model(lin_model, X_train, y_train, X_val, y_val, model_name=f'linear_model_snr_{snr}', save_folder=save_folder)

    nonlin_model = nonlinear_model(input_shape=X_train.shape[1:])
    train_model(nonlin_model, X_train, y_train, X_val, y_val, model_name=f'nonlinear_model_snr_{snr}', save_folder=save_folder)

