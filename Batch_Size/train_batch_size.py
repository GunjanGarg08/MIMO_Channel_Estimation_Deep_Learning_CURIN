import os
import numpy as np
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras.callbacks import ModelCheckpoint

# Suppress TensorFlow warnings and force CPU use
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Function to add noise to the labels
def add_label_noise(y, noise_level=0.1):
    noise = noise_level * np.std(y) * np.random.randn(*y.shape)
    return y + noise

# Define linear and non-linear models
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

# Train and save the models
def train_model(model, X_train, y_train, X_val, y_val, batch_size, model_name, save_folder):
    os.makedirs(save_folder, exist_ok=True)
    model_path = os.path.join(save_folder, f'{model_name}_batch{batch_size}.keras')
    
    checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', mode='min')
    model.fit(X_train, y_train, epochs=5, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[checkpoint])
    print(f"Model saved to {model_path}")

# Main function to run training for different batch sizes and SNR levels
if __name__ == "__main__":
    snr_values = [-10, -5, 0, 5, 10, 15, 20]
    batch_sizes = [32, 64, 128]
    save_folder = "/workspaces/final-code/SNR_Variation/Batch_Size/Trained_Model"

    for batch_size in batch_sizes:
        for snr in snr_values:
            print(f"\nTraining models for Batch Size = {batch_size}, SNR = {snr} dB")

            # Load the dataset
            dataset_file = f'/workspaces/final-code/SNR_Variation/Batch_Size/Generated_Data/dataset_batch_{batch_size}_snr_{snr}.npz'
            if not os.path.exists(dataset_file):
                print(f"Dataset not found: {dataset_file}")
                continue
            
            data = np.load(dataset_file)
            X_train, X_val = data['X_train'], data['X_val']
            y_train, y_val = data['y_train'], data['y_val']
            
            # Add noise
            X_train_noisy = X_train + np.random.normal(0, 0.1, X_train.shape)
            X_val_noisy = X_val + np.random.normal(0, 0.1, X_val.shape)
            y_train_noisy = add_label_noise(y_train, noise_level=0.5)
            y_val_noisy = add_label_noise(y_val, noise_level=0.5)
            
            # Train Linear Model (FCNN)
            lin_model = linear_model(input_shape=X_train.shape[1:])
            train_model(lin_model, X_train_noisy, y_train_noisy, X_val_noisy, y_val_noisy,
                        batch_size, model_name=f'linear_model_snr{snr}', save_folder=save_folder)
            
            # Train Nonlinear Model (CNN)
            nonlin_model = nonlinear_model(input_shape=X_train.shape[1:])
            train_model(nonlin_model, X_train_noisy, y_train_noisy, X_val_noisy, y_val_noisy,
                        batch_size, model_name=f'nonlinear_model_snr{snr}', save_folder=save_folder)


# import os
# import numpy as np
# from tensorflow.keras import models, layers, regularizers
# from tensorflow.keras.callbacks import ModelCheckpoint

# # Suppress TensorFlow warnings and force CPU use
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# # Function to add noise to the labels
# def add_label_noise(y, noise_level=0.1):
#     noise = noise_level * np.std(y) * np.random.randn(*y.shape)
#     return y + noise

# # Define linear and non-linear models
# def linear_model(input_shape):
#     model = models.Sequential([
#         layers.Flatten(input_shape=input_shape),
#         layers.Dense(256, activation='linear', kernel_regularizer=regularizers.l2(0.01)),
#         layers.Dropout(0.5),
#         layers.Dense(np.prod(input_shape), activation='linear'),
#         layers.Reshape(input_shape)
#     ])
#     model.compile(optimizer='SGD', loss='mse', metrics=['mae'])
#     return model

# def nonlinear_model(input_shape):
#     model = models.Sequential([
#         layers.Reshape((input_shape[0], input_shape[1], 1), input_shape=input_shape),
#         layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
#         layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
#         layers.Flatten(),
#         layers.Dense(256, activation='relu'),
#         layers.Dense(np.prod(input_shape), activation='linear'),
#         layers.Reshape(input_shape)
#     ])
#     model.compile(optimizer='SGD', loss='mse', metrics=['mae'])
#     return model

# # Train and save the models
# def train_model(model, X_train, y_train, X_val, y_val, batch_size, model_name, save_folder):
#     os.makedirs(save_folder, exist_ok=True)
#     model_path = os.path.join(save_folder, f'{model_name}_batch{batch_size}.keras')
    
#     checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', mode='min')
#     model.fit(X_train, y_train, epochs=5, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[checkpoint])
#     print(f"Model saved to {model_path}")

# # Main function to run training for different SNR levels and batch sizes
# if __name__ == "__main__":
#     snr_values = [-10, -5, 0, 5, 10, 15, 20]
#     batch_sizes = [32, 64, 128]  # Different batch sizes
#     save_folder = "/workspaces/final-code/SNR_Variation/Batch_Size/Trained_Model"

#     for batch_size in batch_sizes:
#         print(f"\nTraining models for Batch Size = {batch_size}")
        
#         for snr in snr_values:
#             print(f"  SNR = {snr} dB")

#             # Load the dataset
#             dataset_file = f'/workspaces/final-code/SNR_Variation/Users/Generated_Data/dataset_users_{users}_snr_{snr}.npz'
#             data = np.load(dataset_file)
#             X_train, X_val = data['X_train'], data['X_val']
#             y_train, y_val = data['y_train'], data['y_val']
            
#             # Add noise
#             X_train_noisy = X_train + np.random.normal(0, 0.1, X_train.shape)
#             X_val_noisy = X_val + np.random.normal(0, 0.1, X_val.shape)
#             y_train_noisy = add_label_noise(y_train, noise_level=0.5)
#             y_val_noisy = add_label_noise(y_val, noise_level=0.5)
            
#             # Train Linear Model (FCNN)
#             lin_model = linear_model(input_shape=X_train.shape[1:])
#             train_model(lin_model, X_train_noisy, y_train_noisy, X_val_noisy, y_val_noisy,
#                         batch_size, model_name=f'linear_model_snr{snr}', save_folder=save_folder)
            
#             # Train Nonlinear Model (CNN)
#             nonlin_model = nonlinear_model(input_shape=X_train.shape[1:])
#             train_model(nonlin_model, X_train_noisy, y_train_noisy, X_val_noisy, y_val_noisy,
#                         batch_size, model_name=f'nonlinear_model_snr{snr}', save_folder=save_folder)

# import os
# import numpy as np
# from tensorflow.keras import models, layers, regularizers
# from tensorflow.keras.callbacks import ModelCheckpoint

# # Suppress TensorFlow warnings and force CPU use
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# # Function to add noise to the labels
# def add_label_noise(y, noise_level=0.1):
#     noise = noise_level * np.std(y) * np.random.randn(*y.shape)
#     return y + noise

# # Define linear and non-linear models
# def linear_model(input_shape):
#     model = models.Sequential([
#         layers.Flatten(input_shape=input_shape),
#         layers.Dense(256, activation='linear', kernel_regularizer=regularizers.l2(0.01)),
#         layers.Dropout(0.5),
#         layers.Dense(np.prod(input_shape), activation='linear'),
#         layers.Reshape(input_shape)
#     ])
#     model.compile(optimizer='SGD', loss='mse', metrics=['mae'])
#     return model

# def nonlinear_model(input_shape):
#     model = models.Sequential([
#         layers.Reshape((input_shape[0], input_shape[1], 1), input_shape=input_shape),
#         layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
#         layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
#         layers.Flatten(),
#         layers.Dense(256, activation='relu'),
#         layers.Dense(np.prod(input_shape), activation='linear'),
#         layers.Reshape(input_shape)
#     ])
#     model.compile(optimizer='SGD', loss='mse', metrics=['mae'])
#     return model

# # Train and save the models
# def train_model(model, X_train, y_train, X_val, y_val, batch_size, model_name, save_folder):
#     os.makedirs(save_folder, exist_ok=True)
#     model_path = os.path.join(save_folder, f'{model_name}_batch{batch_size}.keras')
    
#     checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', mode='min')
#     model.fit(X_train, y_train, epochs=5, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[checkpoint])
#     print(f"Model saved to {model_path}")

# # Main function to train models with varying batch sizes and SNRs
# if __name__ == "__main__":
#     snr_values = [-10, -5, 0, 5, 10, 15, 20]
#     batch_sizes = [32, 64, 128]
#     users_list = [2, 4, 6, 8]  # User counts
#     save_folder = "/workspaces/final-code/SNR_Variation/Batch_Size/Trained_Model"

#     for users in users_list:
#         print(f"\nTraining models for Users = {users}")
        
#         for batch_size in batch_sizes:
#             print(f"  Batch Size = {batch_size}")
            
#             for snr in snr_values:
#                 print(f"    SNR = {snr} dB")
                
#                 # Correct file path based on users and snr
#                 dataset_file = f'/workspaces/final-code/SNR_Variation/Users/Generated_Data/dataset_users_{users}_snr_{snr}.npz'
                
#                 if not os.path.exists(dataset_file):
#                     print(f"Dataset file not found: {dataset_file}")
#                     continue
                
#                 # Load dataset
#                 data = np.load(dataset_file)
#                 X_train, X_val = data['X_train'], data['X_val']
#                 y_train, y_val = data['y_train'], data['y_val']
                
#                 # Add noise
#                 X_train_noisy = X_train + np.random.normal(0, 0.1, X_train.shape)
#                 X_val_noisy = X_val + np.random.normal(0, 0.1, X_val.shape)
#                 y_train_noisy = add_label_noise(y_train, noise_level=0.5)
#                 y_val_noisy = add_label_noise(y_val, noise_level=0.5)
                
#                 # Train Linear Model (FCNN)
#                 lin_model = linear_model(input_shape=X_train.shape[1:])
#                 train_model(lin_model, X_train_noisy, y_train_noisy, X_val_noisy, y_val_noisy,
#                             batch_size, model_name=f'linear_users{users}_snr{snr}', save_folder=save_folder)
                
#                 # Train Nonlinear Model (CNN)
#                 nonlin_model = nonlinear_model(input_shape=X_train.shape[1:])
#                 train_model(nonlin_model, X_train_noisy, y_train_noisy, X_val_noisy, y_val_noisy,
#                             batch_size, model_name=f'nonlinear_users{users}_snr{snr}', save_folder=save_folder)