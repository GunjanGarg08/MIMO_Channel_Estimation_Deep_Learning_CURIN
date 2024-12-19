import os
import numpy as np
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras.callbacks import ModelCheckpoint

# Suppress TensorFlow warnings and force CPU use
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Only show errors

# Function to add noise to the labels
def add_label_noise(y, noise_level=0.1):
    noise = noise_level * np.std(y) * np.random.randn(*y.shape)
    return y + noise

# Define Linear Model
def linear_model(input_shape):
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=input_shape))
    model.add(layers.Dense(256, activation='linear', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(np.prod(input_shape), activation='linear', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.Reshape(input_shape))
    model.compile(optimizer='SGD', loss='mse', metrics=['mae'])
    return model

# Define Non-linear (CNN) Model
def nonlinear_model(input_shape):
    model = models.Sequential()
    model.add(layers.Reshape((input_shape[0], input_shape[1], 1), input_shape=input_shape))
    model.add(layers.Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(np.prod(input_shape), activation='linear', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.Reshape(input_shape))
    model.compile(optimizer='SGD', loss='mse', metrics=['mae'])
    return model

# Train and Save Models
def train_model(model, X_train, y_train, X_val, y_val, model_name='model', save_folder='./Trained_Models'):
    os.makedirs(save_folder, exist_ok=True)
    best_weights_path = os.path.join(save_folder, f'{model_name}_best_weights.keras')
    final_model_path = os.path.join(save_folder, f'{model_name}_final_model.keras')

    checkpoint = ModelCheckpoint(best_weights_path, save_best_only=True, monitor='val_loss', mode='min')
    history = model.fit(X_train, y_train, epochs=3, batch_size=64, validation_data=(X_val, y_val), callbacks=[checkpoint])
    model.save(final_model_path)
    print(f"Model saved to: {final_model_path}")
    return history

# Main Function for Training across Users and SNR
if __name__ == "__main__":
    snr_values = [-10, -5, 0, 5, 10, 15, 20]
    user_counts = [2, 4, 6, 8]
    noise_factor = 0.5
    dataset_folder = "/workspaces/MIMO_Channel_Estimation_Deep_Learning_CURIN/Users/Generated_Data"
    save_folder = "/workspaces/MIMO_Channel_Estimation_Deep_Learning_CURIN/Users/Trained_Model"

    for users in user_counts:
        print(f"\n--- Training Models for {users} Users ---")
        for snr in snr_values:
            print(f"  SNR: {snr} dB")
            
            # Load dataset
            dataset_path = os.path.join(dataset_folder, f'dataset_users_{users}_snr_{snr}.npz')
            if not os.path.exists(dataset_path):
                print(f"    Dataset not found: {dataset_path}")
                continue

            data = np.load(dataset_path)
            X_train, X_val, X_test = data['X_train'], data['X_val'], data['X_test']
            y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']

            # Add noise to input and labels
            X_train_noisy = X_train + noise_factor * np.random.randn(*X_train.shape)
            X_val_noisy = X_val + noise_factor * np.random.randn(*X_val.shape)
            y_train_noisy = add_label_noise(y_train, noise_level=0.5)
            y_val_noisy = add_label_noise(y_val, noise_level=0.5)

            # Train Linear Model
            lin_model = linear_model(input_shape=X_train.shape[1:])
            lin_model_name = f'linear_model_users_{users}_snr_{snr}'
            train_model(lin_model, X_train_noisy, y_train_noisy, X_val_noisy, y_val_noisy, model_name=lin_model_name, save_folder=save_folder)

            # Train Nonlinear Model
            nonlin_model = nonlinear_model(input_shape=X_train.shape[1:])
            nonlin_model_name = f'nonlinear_model_users_{users}_snr_{snr}'
            train_model(nonlin_model, X_train_noisy, y_train_noisy, X_val_noisy, y_val_noisy, model_name=nonlin_model_name, save_folder=save_folder)

    print("\nTraining completed for all users and SNR levels.")