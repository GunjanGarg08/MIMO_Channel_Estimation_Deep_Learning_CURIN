import os
import numpy as np
from sklearn.model_selection import train_test_split

# Function to generate THz MIMO dataset with variation of batch sizes
def generate_thz_mimo_data_for_batch_sizes(num_samples=10000, num_antennas=256, num_users=2,
                                           snr_values=None, batch_sizes=None, save_folder=None):
    if snr_values is None:
        snr_values = [-10, -5, 0, 5, 10, 15, 20]
    if batch_sizes is None:
        batch_sizes = [32, 64, 128]
    if save_folder is None:
        save_folder = "/workspaces/final-code/SNR_Variation/Batch_Size/Generated_Data"
    
    os.makedirs(save_folder, exist_ok=True)
    
    # Generate the clean dataset (LoS + NLoS components)
    los_component = np.random.randn(num_samples, num_antennas, num_users)
    nlos_component = np.random.randn(num_samples, num_antennas, num_users) * 0.1
    channel_matrix = los_component + nlos_component

    # Generate datasets for each SNR and batch size
    for snr in snr_values:
        noise_power = 1 / (10 ** (snr / 10))  # Convert SNR (dB) to noise power
        for batch_size in batch_sizes:
            noise = np.sqrt(noise_power) * np.random.randn(num_samples, num_antennas, num_users)
            noisy_channel_matrix = channel_matrix + noise

            # Split the dataset
            X_train, X_temp, y_train, y_temp = train_test_split(noisy_channel_matrix, channel_matrix, test_size=0.3, random_state=42)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

            # Save the dataset
            filename = os.path.join(save_folder, f'dataset_batch_{batch_size}_snr_{snr}.npz')
            np.savez_compressed(filename, X_train=X_train, X_val=X_val, X_test=X_test, y_train=y_train, y_val=y_val, y_test=y_test)
            print(f"Dataset saved: {filename}")

# Example usage
snr_values = [-10, -5, 0, 5, 10, 15, 20]
batch_sizes = [32, 64, 128]
generate_thz_mimo_data_for_batch_sizes(snr_values=snr_values, batch_sizes=batch_sizes)