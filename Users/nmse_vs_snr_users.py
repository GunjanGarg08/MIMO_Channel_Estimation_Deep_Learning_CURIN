import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error

# Function to compute NMSE (Normalized Mean Squared Error)
def compute_nmse(y_true, y_pred):
    mse = mean_squared_error(y_true.flatten(), y_pred.flatten())
    norm_factor = np.linalg.norm(y_true.flatten()) ** 2
    return 10 * np.log10(mse / norm_factor)

# Function to save results to an Excel file
def save_results_to_excel(user_count, snr_levels, nmse_linear, nmse_nonlinear, filename):
    df = pd.DataFrame({
        'SNR (dB)': snr_levels,
        'Linear Model NMSE (dB)': nmse_linear,
        'Nonlinear Model NMSE (dB)': nmse_nonlinear
    })
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_excel(filename, index=False)
    print(f"Results for {user_count} users saved to '{filename}'")

# SNR levels and user counts to evaluate
snr_levels = [-10, -5, 0, 5, 10, 15, 20]
user_counts = [2, 4, 6, 8]

# Directory Paths
dataset_folder = "/workspaces/final-code/SNR_Variation/Users/Generated_Data"
model_folder = "/workspaces/final-code/SNR_Variation/Users/Trained_Model"
results_folder = "/workspaces/final-code/SNR_Variation/Users/Results"

# Iterate through each user count and SNR level
print("Evaluating models for different user counts and SNR levels:")
for users in user_counts:
    print(f"\n--- Testing for {users} Users ---")
    nmse_linear_all = []
    nmse_nonlinear_all = []
    valid_snr_levels = []

    for snr in snr_levels:
        print(f"  SNR: {snr} dB")
        
        # Load the dataset
        dataset_path = os.path.join(dataset_folder, f'dataset_users_{users}_snr_{snr}.npz')
        if not os.path.exists(dataset_path):
            print(f"    Dataset not found: {dataset_path}")
            continue
        
        data = np.load(dataset_path)
        X_test = data['X_test']
        y_test = data['y_test']
        
        # Load the trained models
        linear_model_path = os.path.join(model_folder, f'linear_model_users_{users}_snr_{snr}_final_model.keras')
        nonlinear_model_path = os.path.join(model_folder, f'nonlinear_model_users_{users}_snr_{snr}_final_model.keras')

        if os.path.exists(linear_model_path) and os.path.exists(nonlinear_model_path):
            linear_model = load_model(linear_model_path)
            nonlinear_model = load_model(nonlinear_model_path)
        else:
            print(f"    Model files not found for SNR {snr} dB")
            continue
        
        # Predict and compute NMSE
        y_pred_linear = linear_model.predict(X_test)
        y_pred_nonlinear = nonlinear_model.predict(X_test)
        
        nmse_linear_value = compute_nmse(y_test, y_pred_linear)
        nmse_nonlinear_value = compute_nmse(y_test, y_pred_nonlinear)

        nmse_linear_all.append(nmse_linear_value)
        nmse_nonlinear_all.append(nmse_nonlinear_value)
        valid_snr_levels.append(snr)

        # Print NMSE results
        print(f"    Linear Model NMSE: {nmse_linear_value:.2f} dB")
        print(f"    Nonlinear Model NMSE: {nmse_nonlinear_value:.2f} dB")

    # Save results to Excel
    result_file = os.path.join(results_folder, f'nmse_vs_snr_users_{users}.xlsx')
    save_results_to_excel(users, valid_snr_levels, nmse_linear_all, nmse_nonlinear_all, result_file)

    # Plot NMSE vs SNR for Linear and Nonlinear Models
    plt.figure()
    plt.plot(valid_snr_levels, nmse_linear_all, 'o-', label='Linear Model NMSE')
    plt.plot(valid_snr_levels, nmse_nonlinear_all, 's-', label='Nonlinear Model NMSE')
    plt.xlabel('SNR (dB)')
    plt.ylabel('NMSE (dB)')
    plt.title(f'NMSE vs SNR for {users} Users')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plot_file = os.path.join(results_folder, f'nmse_vs_snr_users_{users}.png')
    plt.savefig(plot_file)
    plt.show()
    print(f"Plot saved as '{plot_file}'")

print("\nEvaluation completed for all user counts and SNR levels.")