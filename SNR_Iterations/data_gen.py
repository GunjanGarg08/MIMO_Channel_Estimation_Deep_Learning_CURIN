import numpy as np
from sklearn.model_selection import train_test_split
import os

# Function to generate THz MIMO data with specified SNRs
def generate_thz_mimo_data_for_snr(num_samples=10000, num_antennas=256, num_users=10, snr_values=None, save_file=False):
    if snr_values is None:
        snr_values = [-10, -5, 0, 5, 10, 15, 20, 25]
    
    # Generate the clean dataset (LoS + NLoS components)
    los_component = np.random.randn(num_samples, num_antennas, num_users)
    nlos_component = np.random.randn(num_samples, num_antennas, num_users) * 0.1
    channel_matrix = los_component + nlos_component

    # Container to store datasets for different SNR values
    datasets = {}

    # Generate datasets for each SNR value
    for snr in snr_values:
        noise_power = 1 / (10 ** (snr / 10))  # Convert SNR (dB) to noise power
        noise = np.sqrt(noise_power) * np.random.randn(num_samples, num_antennas, num_users)
        noisy_channel_matrix = channel_matrix + noise

        # Split the data into train, validation, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(noisy_channel_matrix, channel_matrix, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # Store the datasets in a dictionary
        datasets[snr] = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }

        output_folder = "/workspaces/final-code/Parameters_change/SNR_Iterations/Generated_Data/"

        # Optionally save the datasets to separate .npz files
        if save_file:
            filename = os.path.join(output_folder, f'thz_mimo_dataset_snr_{snr}.npz')
            # filename = f'thz_mimo_dataset_snr_{snr}.npz'
            np.savez_compressed(filename, X_train=X_train, X_val=X_val, X_test=X_test, y_train=y_train, y_val=y_val, y_test=y_test)
            print(f"Dataset for SNR {snr} dB saved as '{filename}'")

    return datasets

# Example usage
snr_values = [-10, -5, 0, 5, 10, 15, 20, 25]
datasets = generate_thz_mimo_data_for_snr(snr_values=snr_values, save_file=True)