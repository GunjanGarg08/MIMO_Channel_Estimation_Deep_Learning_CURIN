import os
import numpy as np
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras.callbacks import ModelCheckpoint

# Suppress TensorFlow warnings and force CPU use
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress unnecessary warnings

# Function to add noise to the labels
def add_label_noise(y, noise_level=0.1):
    """Add noise to ground-truth labels."""
    noise = noise_level * np.std(y) * np.random.randn(*y.shape)
    return y + noise

# Define Linear Model (FCNN)
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

# Define Nonlinear Model (CNN)
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

# Function to save trained models
def train_model(model, X_train, y_train, X_val, y_val, batch_size, model_name, save_folder):
    """Train and save the model."""
    os.makedirs(save_folder, exist_ok=True)
    model_path = os.path.join(save_folder, f"{model_name}.keras")

    checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', mode='min')
    model.fit(X_train, y_train, epochs=5, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[checkpoint])
    print(f"Model saved to {model_path}")

# Main function to train for different SNR values and noise scaling factors
if __name__ == "__main__":
    # Define SNR values and noise scaling factors
    snr_values = [-10, -5, 0, 5, 10, 15, 20]
    noise_scaling_factors = [0.5, 1, 2]
    batch_size = 64  # Default batch size
    save_folder = "/workspaces/MIMO_Channel_Estimation_Deep_Learning_CURIN/Noise_Power/Trained_Model"

    # Iterate over SNR values and noise scaling factors
    for snr in snr_values:
        for scaling in noise_scaling_factors:
            print(f"\nTraining for SNR={snr} dB, Noise Scaling Factor={scaling}")

            # Load the dataset (modify the path accordingly)
            dataset_file = f"/workspaces/MIMO_Channel_Estimation_Deep_Learning_CURIN/Noise_Power/Generated_Data/dataset_snr_{snr}_scaling_{scaling}.npz"
            if not os.path.exists(dataset_file):
                print(f"Dataset not found: {dataset_file}")
                continue

            data = np.load(dataset_file)
            X_train, X_val = data['X_train'], data['X_val']
            y_train, y_val = data['y_train'], data['y_val']

            # Add noise to the input data and labels
            X_train_noisy = X_train + np.random.normal(0, scaling, X_train.shape)  # Noise scaling factor applied
            X_val_noisy = X_val + np.random.normal(0, scaling, X_val.shape)
            y_train_noisy = add_label_noise(y_train, noise_level=0.5)
            y_val_noisy = add_label_noise(y_val, noise_level=0.5)

            # Train Linear Model (FCNN)
            linear_model_instance = linear_model(input_shape=X_train.shape[1:])
            train_model(linear_model_instance, X_train_noisy, y_train_noisy, X_val_noisy, y_val_noisy,
                        batch_size, model_name=f"linear_model_snr{snr}_scaling{scaling}", save_folder=save_folder)

            # Train Nonlinear Model (CNN)
            nonlinear_model_instance = nonlinear_model(input_shape=X_train.shape[1:])
            train_model(nonlinear_model_instance, X_train_noisy, y_train_noisy, X_val_noisy, y_val_noisy,
                        batch_size, model_name=f"nonlinear_model_snr{snr}_scaling{scaling}", save_folder=save_folder)
