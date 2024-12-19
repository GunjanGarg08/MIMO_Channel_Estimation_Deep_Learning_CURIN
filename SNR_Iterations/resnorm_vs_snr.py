import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import pandas as pd
import os

# Define a function to compute residual norm (difference between prediction and true)
def compute_residual_norm(y_true, y_pred):
    residual = y_true - y_pred
    norm = np.linalg.norm(residual.flatten())
    return norm

# Initialize SNR levels used during dataset generation
snr_levels = [-10, -5, 0, 5, 10, 15, 20]  # Use the same SNR levels as in train.py
residual_norm_linear = []
residual_norm_nonlinear = []

print("Evaluating residual norms across different SNR levels:")
for snr in snr_levels:
    # Load the dataset for the current SNR level
    data_path = f'thz_mimo_dataset_snr_{snr}.npz'
    if os.path.exists(data_path):
        data = np.load(data_path)
        X_test = data['X_test']
        y_test = data['y_test']
    else:
        print(f"Dataset for SNR = {snr} not found. Skipping...")
        continue
    
    # Load the trained models
    linear_model_path = f'linear_model_snr_{snr}_final_model.keras'
    nonlinear_model_path = f'nonlinear_model_snr_{snr}_final_model.keras'
    
    if os.path.exists(linear_model_path) and os.path.exists(nonlinear_model_path):
        linear_model = load_model(linear_model_path)
        nonlinear_model = load_model(nonlinear_model_path)
        
        # Make predictions using the models
        y_pred_linear = linear_model.predict(X_test)
        y_pred_nonlinear = nonlinear_model.predict(X_test)
        
        # Compute residual norm for both models
        residual_norm_linear.append(compute_residual_norm(y_test, y_pred_linear))
        residual_norm_nonlinear.append(compute_residual_norm(y_test, y_pred_nonlinear))
    else:
        print(f"Model files for SNR = {snr} not found. Skipping...")

# Function to save results to an Excel file
def save_results_to_excel(snr_levels, residual_norm_linear, residual_norm_nonlinear, filename='residual_norm_results.xlsx', sheet_name='Residual Norm Results'):
    df = pd.DataFrame({
        'SNR (dB)': snr_levels,
        'Linear Model Residual Norm': residual_norm_linear,
        'Nonlinear Model Residual Norm': residual_norm_nonlinear
    })
    df.to_excel(filename, index=False, sheet_name=sheet_name)
    print(f"Results saved to '{filename}' in sheet '{sheet_name}'")

# Save the residual norm results to an Excel file
save_results_to_excel(snr_levels, residual_norm_linear, residual_norm_nonlinear)

# Plot the results
plt.plot(snr_levels, residual_norm_linear, 'o-', label='Linear Model Residual Norm')
plt.plot(snr_levels, residual_norm_nonlinear, 's-', label='Nonlinear Model Residual Norm')
plt.xlabel('SNR (dB)')
plt.ylabel('Residual Norm')
plt.legend()
plt.grid(True)
plt.title('Residual Norm vs SNR')

# Save the chart
plt.savefig('residual_norm_vs_snr.png')

# Show the plot
plt.show()

print("Chart saved as 'residual_norm_vs_snr.png'")
