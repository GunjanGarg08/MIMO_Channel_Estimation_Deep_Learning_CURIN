import os
import numpy as np
from sklearn.model_selection import train_test_split

# Function to generate THz MIMO data with noise power scaling
def generate_dataset_noise_scaling(snr_values, noise_scaling_factors, num_samples=10000, num_antennas=256, num_users=10, save_folder=None):
    if save_folder is None:
        save_folder = "/workspaces/MIMO_Channel_Estimation_Deep_Learning_CURIN/Noise_Power/Generated_Data"
    os.makedirs(save_folder, exist_ok=True)
    
    # Clean channel matrix (LoS + NLoS)
    los_component = np.random.randn(num_samples, num_antennas, num_users)
    nlos_component = np.random.randn(num_samples, num_antennas, num_users) * 0.1
    channel_matrix = los_component + nlos_component

    # Generate datasets with noise scaling
    for snr in snr_values:
        for scaling in noise_scaling_factors:
            noise_power = scaling / (10 ** (snr / 10))  # Adjusted noise power
            noise = np.sqrt(noise_power) * np.random.randn(num_samples, num_antennas, num_users)
            noisy_channel_matrix = channel_matrix + noise

            # Train, Validation, Test split
            X_train, X_temp, y_train, y_temp = train_test_split(noisy_channel_matrix, channel_matrix, test_size=0.3, random_state=42)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

            # Save the dataset
            filename = os.path.join(save_folder, f"dataset_snr_{snr}_scaling_{scaling}.npz")
            np.savez_compressed(filename, X_train=X_train, X_val=X_val, X_test=X_test, y_train=y_train, y_val=y_val, y_test=y_test)
            print(f"Dataset saved: {filename}")

# Example usage
snr_values = [-10, -5, 0, 5, 10, 15, 20]
noise_scaling_factors = [0.5, 1, 2]  # Different scaling factors
generate_dataset_noise_scaling(snr_values, noise_scaling_factors)
