import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
import os

# Function to compute NMSE (Normalized Mean Squared Error)
def compute_nmse(y_true, y_pred):
    mse = mean_squared_error(y_true.flatten(), y_pred.flatten())
    norm_factor = np.linalg.norm(y_true.flatten()) ** 2
    return 10 * np.log10(mse / norm_factor)  # NMSE in dB

# Function to save results to an Excel file
def save_results_to_excel(snr_levels, batch_sizes, nmse_linear, nmse_nonlinear, filename):
    df = pd.DataFrame({
        'SNR (dB)': np.repeat(snr_levels, len(batch_sizes)),
        'Batch Size': batch_sizes * len(snr_levels),
        'Linear Model NMSE (dB)': nmse_linear,
        'Nonlinear Model NMSE (dB)': nmse_nonlinear
    })
    df.to_excel(filename, index=False)
    print(f"Results saved to '{filename}'")

# SNR levels and batch sizes
snr_levels = [-10, -5, 0, 5, 10, 15, 20]
batch_sizes = [32, 64, 128]

# Storage for NMSE results
nmse_linear_all = []
nmse_nonlinear_all = []

# Path for dataset and trained models
data_folder = "/workspaces/final-code/SNR_Variation/Batch_Size/Generated_Data"
model_folder = "/workspaces/final-code/SNR_Variation/Batch_Size/Trained_Model"

# Loop through SNR levels and batch sizes
print("Evaluating NMSE for different batch sizes and SNR levels:")
for snr in snr_levels:
    for batch_size in batch_sizes:
        print(f"\nEvaluating for SNR = {snr} dB, Batch Size = {batch_size}")
        
        # Load the dataset
        dataset_file = f"{data_folder}/dataset_batch_{batch_size}_snr_{snr}.npz"
        if not os.path.exists(dataset_file):
            print(f"Dataset not found: {dataset_file}")
            continue
        
        data = np.load(dataset_file)
        X_test = data['X_test']
        y_test = data['y_test']
        
        # Load trained models
        linear_model_path = f"{model_folder}/linear_model_snr{snr}_batch{batch_size}.keras"
        nonlinear_model_path = f"{model_folder}/nonlinear_model_snr{snr}_batch{batch_size}.keras"
        
        if not os.path.exists(linear_model_path) or not os.path.exists(nonlinear_model_path):
            print(f"Models not found for SNR {snr} dB and Batch Size {batch_size}")
            continue
        
        linear_model = load_model(linear_model_path)
        nonlinear_model = load_model(nonlinear_model_path)
        
        # Make predictions
        y_pred_linear = linear_model.predict(X_test)
        y_pred_nonlinear = nonlinear_model.predict(X_test)
        
        # Compute NMSE
        nmse_linear = compute_nmse(y_test, y_pred_linear)
        nmse_nonlinear = compute_nmse(y_test, y_pred_nonlinear)
        nmse_linear_all.append(nmse_linear)
        nmse_nonlinear_all.append(nmse_nonlinear)
        
        # Print results
        print(f"  Linear Model NMSE: {nmse_linear:.2f} dB")
        print(f"  Nonlinear Model NMSE: {nmse_nonlinear:.2f} dB")

# Save results to Excel
output_file = "/workspaces/final-code/SNR_Variation/Batch_Size/Results/nmse_vs_snr_batch_size.xlsx"
save_results_to_excel(snr_levels, batch_sizes, nmse_linear_all, nmse_nonlinear_all, output_file)

# Plot NMSE vs SNR for different batch sizes
plt.figure(figsize=(10, 6))
for i, batch_size in enumerate(batch_sizes):
    linear_nmse = nmse_linear_all[i::len(batch_sizes)]
    nonlinear_nmse = nmse_nonlinear_all[i::len(batch_sizes)]
    plt.plot(snr_levels, linear_nmse, 'o--', label=f'Linear Model (Batch {batch_size})')
    plt.plot(snr_levels, nonlinear_nmse, 's-', label=f'Nonlinear Model (Batch {batch_size})')

plt.xlabel('SNR (dB)')
plt.ylabel('NMSE (dB)')
plt.title('NMSE vs SNR for Different Batch Sizes')
plt.legend()
plt.grid(True)

# Save and show the plot
plot_file = "/workspaces/final-code/SNR_Variation/Batch_Size/Results/nmse_vs_snr_batch_size.png"
plt.savefig(plot_file, bbox_inches='tight')
plt.show()

print(f"Graph saved as '{plot_file}'")