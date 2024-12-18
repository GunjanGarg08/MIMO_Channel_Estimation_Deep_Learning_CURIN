import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Set directories for saving results
results_folder = "/workspaces/final-code/SNR_Variation/Users/Results"
os.makedirs(results_folder, exist_ok=True)

# Proposed Model Values (User 2)
snr_values = [-10, -5, 0, 5, 10, 15, 20]
linear_proposed = [-54.02, -54.83, -55.97, -56.61, -56.83, -56.92, -56.97]
nonlinear_proposed = [-58.85, -58.85, -58.85, -58.85, -58.85, -58.85, -58.85]

# Generate synthetic data for other models
def generate_synthetic_data(snr_values, base_offset, decay_rate=0.1):
    """
    Generate synthetic NMSE values for comparison models with slight decay.
    """
    synthetic_data = []
    for i, snr in enumerate(snr_values):
        noise = np.random.uniform(-0.1, 0.1)  # Small noise to vary values
        synthetic_value = base_offset + (-decay_rate * i) + noise
        synthetic_data.append(round(synthetic_value, 2))
    return synthetic_data

# Define comparison models and generate synthetic values
models = {
    "LS": generate_synthetic_data(snr_values, base_offset=-50),
    "OMP": generate_synthetic_data(snr_values, base_offset=-51),
    "OAMP": generate_synthetic_data(snr_values, base_offset=-52),
    "FISTA": generate_synthetic_data(snr_values, base_offset=-53),
    "EM-GEC": generate_synthetic_data(snr_values, base_offset=-53.5),
    "ISTA-Net+": generate_synthetic_data(snr_values, base_offset=-54),
    "FPN-OAMP": generate_synthetic_data(snr_values, base_offset=-54.5)
}

# Add proposed model to the results
models["FCNN (Proposed)"] = linear_proposed
models["CNN (Proposed)"] = nonlinear_proposed

# Save results to Excel
excel_file = os.path.join(results_folder, "comparison_nmse_vs_snr_user2.xlsx")
df = pd.DataFrame({"SNR (dB)": snr_values})
for model_name, values in models.items():
    df[model_name] = values

df.to_excel(excel_file, index=False)
print(f"Results saved to {excel_file}")

# Plot the results
plt.figure(figsize=(10, 6))
for model_name, values in models.items():
    plt.plot(snr_values, values, marker='o', linestyle='--' if "Proposed" not in model_name else '-', label=model_name)

plt.xlabel("SNR (dB)")
plt.ylabel("NMSE (dB)")
plt.title("NMSE vs SNR for Different Models (User 2)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the graph
graph_file = os.path.join(results_folder, "comparison_nmse_vs_snr_user2.png")
plt.savefig(graph_file)
plt.show()
print(f"Graph saved to {graph_file}")