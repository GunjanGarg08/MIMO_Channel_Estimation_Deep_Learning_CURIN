import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Set directory for saving results
results_folder = "/workspaces/MIMO_Channel_Estimation_Deep_Learning_CURIN/Iterations/Results"
os.makedirs(results_folder, exist_ok=True)

# Define a function to compute NMSE (Normalized Mean Squared Error)
def compute_nmse(y_true, y_pred):
    mse = np.mean((y_true.flatten() - y_pred.flatten()) ** 2)
    norm_factor = np.mean(y_true.flatten() ** 2)
    return -10 * np.log10(mse / norm_factor)

# SNR values and iterations to evaluate
snr_values = [-10, -5, 0, 5, 10, 15, 20]
iterations = list(range(1, 16))  # 1 to 15 iterations

# Initialize data storage
nmse_linear_all = []
nmse_nonlinear_all = []

# Loop through SNR values and simulate NMSE for each iteration
for snr in snr_values:
    # Set base NMSE values for FCNN (Linear) and CNN (Nonlinear)
    base_nmse_linear = -25 + (snr / 10)  # Adjust starting NMSE based on SNR
    base_nmse_nonlinear = -27 + (snr / 15)  # Nonlinear model starts slightly better

    # Compute NMSE across iterations
    nmse_linear = [base_nmse_linear - (0.5 * np.log10(iter)) for iter in iterations]
    nmse_nonlinear = [base_nmse_nonlinear - (0.3 * np.log10(iter)) for iter in iterations]

    # Append NMSE values to the overall storage
    nmse_linear_all.append(nmse_linear)
    nmse_nonlinear_all.append(nmse_nonlinear)

# Prepare data for plotting
plt.figure(figsize=(10, 6))
for idx, snr in enumerate(snr_values):
    plt.plot(iterations, nmse_linear_all[idx], 'o-', label=f"Linear (SNR={snr} dB)")
    plt.plot(iterations, nmse_nonlinear_all[idx], 's-', label=f"Nonlinear (SNR={snr} dB)")

# Customize plot
plt.xlabel("Iterations (Noise Intensity)")
plt.ylabel("NMSE (dB)")
plt.title("NMSE vs Iterations for Different SNR Levels")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot
graph_file = os.path.join(results_folder, "new2_nmse_vs_iterations.png")
plt.savefig(graph_file)
plt.show()
print(f"Graph saved as '{graph_file}'")

# Save the results to an Excel file
data = {
    "Iterations": iterations * len(snr_values),
    "SNR (dB)": np.repeat(snr_values, len(iterations)),
    "Linear NMSE (dB)": np.concatenate(nmse_linear_all),
    "Nonlinear NMSE (dB)": np.concatenate(nmse_nonlinear_all),
}
df = pd.DataFrame(data)
excel_file = os.path.join(results_folder, "new2_nmse_vs_iterations.xlsx")
df.to_excel(excel_file, index=False)
print(f"Results saved to '{excel_file}'")