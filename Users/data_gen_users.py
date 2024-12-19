import numpy as np
from sklearn.model_selection import train_test_split
import os

# Function to generate THz MIMO data for different users and SNRs
def generate_thz_mimo_data_for_users_and_snr(num_samples=10000, num_antennas=256, snr_values=None, user_counts=None, save_file=False):
    if snr_values is None:
        snr_values = [-10, -5, 0, 5, 10, 15, 20]  # SNR range
    if user_counts is None:
        user_counts = [2, 4, 6, 8]  # Different numbers of users
    
    output_folder = "/workspaces/MIMO_Channel_Estimation_Deep_Learning_CURIN/Users/Generated_Data"
    os.makedirs(output_folder, exist_ok=True)

    for users in user_counts:
        print(f"\nGenerating data for {users} users:")
        for snr in snr_values:
            noise_power = 1 / (10 ** (snr / 10))
            los_component = np.random.randn(num_samples, num_antennas, users)
            nlos_component = np.random.randn(num_samples, num_antennas, users) * 0.1
            channel_matrix = los_component + nlos_component
            noise = np.sqrt(noise_power) * np.random.randn(num_samples, num_antennas, users)
            noisy_channel_matrix = channel_matrix + noise

            # Split the data
            X_train, X_temp, y_train, y_temp = train_test_split(noisy_channel_matrix, channel_matrix, test_size=0.3, random_state=42)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

            # Save the data
            filename = os.path.join(output_folder, f'dataset_users_{users}_snr_{snr}.npz')
            np.savez_compressed(filename, X_train=X_train, X_val=X_val, X_test=X_test, y_train=y_train, y_val=y_val, y_test=y_test)
            print(f"  Saved dataset for SNR {snr} dB and {users} users: '{filename}'")

# Run the data generation
snr_values = [-10, -5, 0, 5, 10, 15, 20]
user_counts = [2, 4, 6, 8]
generate_thz_mimo_data_for_users_and_snr(snr_values=snr_values, user_counts=user_counts, save_file=True)