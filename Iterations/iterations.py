import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import load_model

def compute_nmse(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    norm_factor = np.mean(y_true ** 2)
    return -10 * np.log10(mse / norm_factor)

snr_values = [-10, -5, 0, 5, 10, 15, 20]
iterations = list(range(1, 16))
nmse_linear_all, nmse_nonlinear_all = [], []

for snr in snr_values:
    dataset_path = f"/workspaces/MIMO_Channel_Estimation_Deep_Learning_CURIN/Iterations/Generated_Data/thz_mimo_dataset_snr_{snr}.npz"
    data = np.load(dataset_path)
    X_test, y_test = data['X_test'], data['y_test']

    linear_model_path = f"/workspaces/MIMO_Channel_Estimation_Deep_Learning_CURIN/Iterations/Trained_Model/linear_model_snr_{snr}.keras"
    nonlinear_model_path = f"/workspaces/MIMO_Channel_Estimation_Deep_Learning_CURIN/Iterations/Trained_Model/nonlinear_model_snr_{snr}.keras"

    linear_model = load_model(linear_model_path)
    nonlinear_model = load_model(nonlinear_model_path)

    nmse_linear, nmse_nonlinear = [], []

    for iteration in iterations:
        noise_power = 10 ** (-snr / 10)
        noise = np.sqrt(noise_power) * np.random.randn(*X_test.shape) * iteration
        X_test_noisy = X_test + noise

        y_pred_linear = linear_model.predict(X_test_noisy)
        y_pred_nonlinear = nonlinear_model.predict(X_test_noisy)

        nmse_linear.append(compute_nmse(y_test, y_pred_linear))
        nmse_nonlinear.append(compute_nmse(y_test, y_pred_nonlinear))

    nmse_linear_all.append(nmse_linear)
    nmse_nonlinear_all.append(nmse_nonlinear)

    plt.plot(iterations, nmse_linear, label=f'Linear (SNR={snr} dB)')
    plt.plot(iterations, nmse_nonlinear, label=f'Nonlinear (SNR={snr} dB)')

plt.xlabel('Iterations')
plt.ylabel('NMSE (dB)')
plt.title('NMSE vs Iterations for Different SNRs')
plt.legend()
plt.grid(True)
plt.savefig("/workspaces/MIMO_Channel_Estimation_Deep_Learning_CURIN/Iterations/Results/nmse_vs_iterations.png")
plt.show()

results_df = pd.DataFrame({
    'SNR': np.repeat(snr_values, len(iterations)),
    'Iteration': iterations * len(snr_values),
    'Linear NMSE': np.concatenate(nmse_linear_all),
    'Nonlinear NMSE': np.concatenate(nmse_nonlinear_all)
})

results_df.to_excel("/workspaces/MIMO_Channel_Estimation_Deep_Learning_CURIN/Iterations/Results/nmse_vs_iterations.xlsx", index=False)
print("Results saved to 'nmse_vs_iterations_results.xlsx'")

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from tensorflow.keras.models import load_model
# from sklearn.metrics import mean_squared_error
# import os

# # Define a function to compute NMSE (Normalized Mean Squared Error)
# def compute_nmse(y_true, y_pred):
#     mse = mean_squared_error(y_true.flatten(), y_pred.flatten())
#     norm_factor = np.linalg.norm(y_true.flatten()) ** 2
#     return -10 * np.log10(mse / norm_factor)  # Ensuring values are in dB range

# # Define a function to save NMSE results to an Excel file
# def save_results_to_excel(snr_values, iterations, nmse_linear, nmse_nonlinear, filename):
#     df = pd.DataFrame({
#         'SNR (dB)': np.repeat(snr_values, len(iterations)),
#         'Iterations': iterations * len(snr_values),
#         'Linear Model NMSE (dB)': nmse_linear,
#         'Nonlinear Model NMSE (dB)': nmse_nonlinear
#     })
#     df.to_excel(filename, index=False)
#     print(f"Results saved to '{filename}'")

# # SNR values and iterations to evaluate
# snr_values = [-10, -5, 0, 5, 10, 15, 20]
# iterations = list(range(1, 16))  # 1 to 15 iterations

# # Store NMSE values for all SNRs and iterations
# nmse_linear_all = []
# nmse_nonlinear_all = []

# print("Evaluating models across different SNR values and iterations:")
# for snr in snr_values:
#     print(f"\nEvaluating for SNR: {snr} dB")

#     # Load the dataset for the current SNR
#     data = np.load(f'/workspaces/MIMO_Channel_Estimation_Deep_Learning_CURIN/Iterations/Generated_Data/thz_mimo_dataset_snr_{snr}.npz')
#     X_test = data['X_test']
#     y_test = data['y_test']

#     # Load the trained models for the current SNR
#     linear_model_path = f'/workspaces/MIMO_Channel_Estimation_Deep_Learning_CURIN/Iterations/Trained_Model/linear_model_snr_{snr}.keras'
#     nonlinear_model_path = f'/workspaces/MIMO_Channel_Estimation_Deep_Learning_CURIN/Iterations/Trained_Model/nonlinear_model_snr_{snr}.keras'

#     if not os.path.exists(linear_model_path) or not os.path.exists(nonlinear_model_path):
#         print(f"Model files for SNR {snr} dB not found.")
#         continue

#     linear_model = load_model(linear_model_path)
#     nonlinear_model = load_model(nonlinear_model_path)

#     # Initialize lists to store NMSE values for each iteration for this SNR
#     nmse_linear = []
#     nmse_nonlinear = []

#     for iteration in iterations:
#         # Add noise according to the iteration count and SNR
#         noise_power = 10 ** (-snr / 10)
#         noise = np.random.normal(0, np.sqrt(noise_power / 10), X_test.shape) * iteration  # Adjusted scaling
#         X_test_noisy = X_test + noise  # Adding noise to the test data

#         # Make predictions
#         y_pred_linear = linear_model.predict(X_test_noisy)
#         y_pred_nonlinear = nonlinear_model.predict(X_test_noisy)

#         # Compute NMSE for both models
#         nmse_linear_value = compute_nmse(y_test, y_pred_linear)
#         nmse_nonlinear_value = compute_nmse(y_test, y_pred_nonlinear)
#         nmse_linear.append(nmse_linear_value)
#         nmse_nonlinear.append(nmse_nonlinear_value)

#         # Print NMSE results for the current iteration
#         print(f"  Iteration: {iteration}")
#         print(f"    Linear Model NMSE: {nmse_linear_value:.2f} dB")
#         print(f"    Nonlinear Model NMSE: {nmse_nonlinear_value:.2f} dB")

#     # Append the results for this SNR
#     nmse_linear_all.extend(nmse_linear)
#     nmse_nonlinear_all.extend(nmse_nonlinear)

#     # Plot NMSE for this SNR
#     plt.plot(iterations, nmse_linear, 'o-', label=f'Linear (SNR={snr} dB)')
#     plt.plot(iterations, nmse_nonlinear, 's-', label=f'Nonlinear (SNR={snr} dB)')

# # Configure the plot
# plt.xlabel('Iterations (Noise Intensity)')
# plt.ylabel('NMSE (dB)')
# plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
# plt.grid(True)
# plt.title('NMSE vs Iterations for Different SNR Levels')

# # Save the plot
# plot_filename = '/workspaces/MIMO_Channel_Estimation_Deep_Learning_CURIN/Iterations/Results/new_nmse_vs_iterations.png'
# plt.savefig(plot_filename, bbox_inches='tight')
# plt.show()
# print(f"Chart saved as '{plot_filename}'")

# # Save the NMSE results to an Excel file
# excel_filename = '/workspaces/MIMO_Channel_Estimation_Deep_Learning_CURIN/Iterations/Results/new_nmse_vs_iterations.xlsx'
# save_results_to_excel(snr_values, iterations, nmse_linear_all, nmse_nonlinear_all, excel_filename)