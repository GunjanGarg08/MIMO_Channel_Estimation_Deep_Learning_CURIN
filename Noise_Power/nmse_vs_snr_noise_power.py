import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error

# Function to compute NMSE
def compute_nmse(y_true, y_pred):
    mse = mean_squared_error(y_true.flatten(), y_pred.flatten())
    norm_factor = np.linalg.norm(y_true.flatten()) ** 2
    return 5 * np.log10(mse / norm_factor)

# Parameters
snr_values = [-10, -5, 0, 5, 10, 15, 20]
noise_scaling_factors = [0.5, 1, 2]
save_folder = "/workspaces/MIMO_Channel_Estimation_Deep_Learning_CURIN/Noise_Power/Trained_Model"
results = []

# Evaluate models
for snr in snr_values:
    for scaling in noise_scaling_factors:
        dataset_file = f"/workspaces/MIMO_Channel_Estimation_Deep_Learning_CURIN/Noise_Power/Generated_Data/dataset_snr_{snr}_scaling_{scaling}.npz"
        if not os.path.exists(dataset_file):
            print(f"Dataset missing: {dataset_file}")
            continue
        
        data = np.load(dataset_file)
        X_test, y_test = data['X_test'], data['y_test']
        
        # Load models
        linear_model_path = os.path.join(save_folder, f"linear_model_snr{snr}_scaling{scaling}.keras")
        nonlinear_model_path = os.path.join(save_folder, f"nonlinear_model_snr{snr}_scaling{scaling}.keras")
        
        if os.path.exists(linear_model_path) and os.path.exists(nonlinear_model_path):
            lin_model = load_model(linear_model_path)
            nonlin_model = load_model(nonlinear_model_path)

            # Predictions
            y_pred_lin = lin_model.predict(X_test)
            y_pred_nonlin = nonlin_model.predict(X_test)

            # Compute NMSE
            nmse_lin = compute_nmse(y_test, y_pred_lin)
            nmse_nonlin = compute_nmse(y_test, y_pred_nonlin)
            results.append([snr, scaling, nmse_lin, nmse_nonlin])
        else:
            print(f"Model files missing for SNR={snr}, Scaling={scaling}")

# Save results to Excel
df = pd.DataFrame(results, columns=['SNR (dB)', 'Noise Scaling', 'Linear Model NMSE', 'Nonlinear Model NMSE'])
df.to_excel("/workspaces/MIMO_Channel_Estimation_Deep_Learning_CURIN/Noise_Power/Results/new_nmse_vs_snr_noise_scaling.xlsx", index=False)

# Plot the results
plt.figure(figsize=(10, 6))
for scaling in noise_scaling_factors:
    subset = df[df['Noise Scaling'] == scaling]
    plt.plot(subset['SNR (dB)'], subset['Linear Model NMSE'], label=f"Linear Model (Scaling={scaling})", marker='o')
    plt.plot(subset['SNR (dB)'], subset['Nonlinear Model NMSE'], label=f"Nonlinear Model (Scaling={scaling})", marker='s')

plt.xlabel("SNR (dB)")
plt.ylabel("NMSE (dB)")
plt.title("NMSE vs SNR for Different Noise Power Scaling")
plt.legend()
plt.grid()
plt.savefig("/workspaces/MIMO_Channel_Estimation_Deep_Learning_CURIN/Noise_Power/Results/new_nmse_vs_snr_noise_scaling.png")
plt.show()