import numpy as np
from sklearn.model_selection import train_test_split
import os

def generate_thz_mimo_data_for_snr(num_samples=10000, num_antennas=256, num_users=10, snr_values=None, save_file=False):
    if snr_values is None:
        snr_values = [-10, -5, 0, 5, 10, 15, 20]

    los_component = np.random.randn(num_samples, num_antennas, num_users)
    nlos_component = np.random.randn(num_samples, num_antennas, num_users) * 0.1
    channel_matrix = los_component + nlos_component

    datasets = {}

    for snr in snr_values:
        noise_power = 1 / (10 ** (snr / 10))
        noise = np.sqrt(noise_power) * np.random.randn(num_samples, num_antennas, num_users)
        noisy_channel_matrix = channel_matrix + noise

        X_train, X_temp, y_train, y_temp = train_test_split(noisy_channel_matrix, channel_matrix, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        datasets[snr] = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }

        if save_file:
            output_folder = "/workspaces/MIMO_Channel_Estimation_Deep_Learning_CURIN/Iterations/Generated_Data"
            os.makedirs(output_folder, exist_ok=True)
            filename = os.path.join(output_folder, f'thz_mimo_dataset_snr_{snr}.npz')
            np.savez_compressed(filename, X_train=X_train, X_val=X_val, X_test=X_test, y_train=y_train, y_val=y_val, y_test=y_test)
            print(f"Dataset for SNR {snr} dB saved as '{filename}'")

    return datasets

snr_values = [-10, -5, 0, 5, 10, 15, 20]
datasets = generate_thz_mimo_data_for_snr(snr_values=snr_values, save_file=True)