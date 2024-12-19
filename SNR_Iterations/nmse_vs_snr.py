import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
import os

# Define a function to compute NMSE (Normalized Mean Squared Error)
def compute_nmse(y_true, y_pred):
    mse = mean_squared_error(y_true.flatten(), y_pred.flatten())
    norm_factor = np.linalg.norm(y_true.flatten()) ** 2
    return 5 * np.log10(mse / norm_factor)

# Define a function to save results to an Excel file
def save_results_to_excel(snr_levels, nmse_linear, nmse_nonlinear, filename='/workspaces/final-code/Parameters_change/SNR_Iterations/nmse_vs_snr_results.xlsx'):
    df = pd.DataFrame({
        'SNR (dB)': snr_levels,
        'Linear Model NMSE (dB)': nmse_linear,
        'Nonlinear Model NMSE (dB)': nmse_nonlinear
    })
    df.to_excel(filename, index=False)
    print(f"Results saved to '{filename}'")

# List of SNR levels used during dataset generation
snr_levels = [-10, -5, 0, 5, 10, 15, 20]

# Initialize lists to store NMSE results for each model
valid_snr_levels = []  # To store SNR levels for which processing was successful
nmse_linear = []
nmse_nonlinear = []

print("Evaluating models across different SNR levels:")
for snr in snr_levels:
    # Load the dataset for the current SNR level
    data_file = f'/workspaces/final-code/Parameters_change/SNR_Iterations/Generated_Data/thz_mimo_dataset_snr_{snr}.npz'
    if not os.path.exists(data_file):
        print(f"Data file not found: {data_file}")
        continue
    
    data = np.load(data_file)
    X_test = data['X_test']
    y_test = data['y_test']
    
    # Paths to the models for the current SNR level
    linear_model_path = f'/workspaces/final-code/Parameters_change/SNR_Iterations/Trained_Models/linear_model_snr_{snr}_final_model.keras'
    nonlinear_model_path = f'/workspaces/final-code/Parameters_change/SNR_Iterations/Trained_Models/nonlinear_model_snr_{snr}_final_model.keras'

    # Load models if available, else skip to next SNR level
    if os.path.exists(linear_model_path) and os.path.exists(nonlinear_model_path):
        linear_model = load_model(linear_model_path)
        nonlinear_model = load_model(nonlinear_model_path)
    else:
        print(f"Model files not found for SNR {snr} dB")
        continue
    
    # Make predictions using the models
    y_pred_linear = linear_model.predict(X_test)
    y_pred_nonlinear = nonlinear_model.predict(X_test)
    
    # Compute NMSE for both models
    nmse_linear_value = compute_nmse(y_test, y_pred_linear)
    nmse_nonlinear_value = compute_nmse(y_test, y_pred_nonlinear)
    nmse_linear.append(nmse_linear_value)
    nmse_nonlinear.append(nmse_nonlinear_value)
    valid_snr_levels.append(snr)  # Record the successfully processed SNR level
    
    # Print NMSE results for the current SNR level
    print(f"SNR: {snr} dB")
    print(f"  Linear Model NMSE: {nmse_linear_value:.2f} dB")
    print(f"  Nonlinear Model NMSE: {nmse_nonlinear_value:.2f} dB")

# Save the NMSE results to an Excel file
save_results_to_excel(valid_snr_levels, nmse_linear, nmse_nonlinear)

# Plot the results
plt.plot(valid_snr_levels, nmse_linear, 'o-', label='Linear Model NMSE')
plt.plot(valid_snr_levels, nmse_nonlinear, 's-', label='Nonlinear Model NMSE')
plt.xlabel('SNR (dB)')
plt.ylabel('NMSE (dB)')
plt.legend()
plt.grid(True)
plt.title('NMSE vs SNR')

# Save the plot
plt.savefig('/workspaces/final-code/Parameters_change/SNR_Iterations/nmse_vs_snr.png')

# Show the plot
plt.show()

print("Chart saved as 'nmse_vs_snr.png'")
