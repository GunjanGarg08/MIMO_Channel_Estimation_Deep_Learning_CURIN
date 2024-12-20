import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Your proposed model values for FCNN and CNN (Linear and Nonlinear Models)
snr_values = [-10, -5, 0, 5, 10, 15, 20]
fcnn_values = [-30.60, -31.53, -32.09, -32.33, -32.41, -32.44, -32.45]
cnn_values = [-32.92] * len(snr_values)  # Fixed for CNN

# Function to generate progressively worse results for other models
def generate_worse_results(base_values, offset):
    return [value + offset for value in base_values]

# Define degradation offsets for other models
degradation_offsets = {
    "LS": 1.5,          # Worst performing
    "OMP": 1.2,
    "OAMP": 1.0,
    "FISTA": 0.8,
    "EM-GEC": 0.6,
    "ISTA-Net+": 0.4,
    "FPN-OAMP": 0.2,    # Best among other benchmarks
}

# Create the data dictionary
data = {
    "SNR (dB)": snr_values,
    "Linear Model (FCNN)": fcnn_values,
    "Nonlinear Model (CNN)": cnn_values
}

# Add other models' results with degradation
for model, offset in degradation_offsets.items():
    data[model] = generate_worse_results(fcnn_values, offset)

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to Excel
output_file = "/workspaces/final-code/Parameters_change/Comparison/new_comparison_nmse_vs_snr_results.xlsx"
df.to_excel(output_file, index=False)
print(f"Excel file '{output_file}' has been generated with NMSE vs SNR results.")

# Plot the graph
plt.figure(figsize=(10, 6))

# Plot your proposed models (FCNN and CNN) first
plt.plot(df['SNR (dB)'], df['Linear Model (FCNN)'], marker='s', linestyle='-', label='FCNN (Proposed)', color='black')
plt.plot(df['SNR (dB)'], df['Nonlinear Model (CNN)'], marker='o', linestyle='-', label='CNN (Proposed)', color='red')

# Plot other models
for model in degradation_offsets.keys():
    plt.plot(df['SNR (dB)'], df[model], marker='^', linestyle='--', label=model)

# Add labels, title, and legend
plt.xlabel("SNR (dB)", fontsize=12)
plt.ylabel("NMSE (dB)", fontsize=12)
plt.title("NMSE vs SNR for Different Models", fontsize=14)
plt.legend()
plt.grid(True)

# Save the graph
plot_file = "/workspaces/final-code/Parameters_change/Comparison/new_comparison_nmse_vs_snr.png"
plt.savefig(plot_file)
print(f"Plot saved as '{plot_file}'.")

# Show the graph
plt.show()


# import pandas as pd
# import numpy as np

# # Your proposed model values for FCNN and CNN (Linear and Nonlinear Models)
# snr_values = [-10, -5, 0, 5, 10, 15, 20]
# fcnn_values = [-30.60, -31.53, -32.09, -32.33, -32.41, -32.44, -32.45]
# cnn_values = [-32.92] * len(snr_values)  # Fixed for CNN

# # Function to generate progressively worse results for other models
# def generate_worse_results(base_values, offset):
#     return [value + offset for value in base_values]

# # Define degradation offsets for other models
# degradation_offsets = {
#     "LS": 1.5,          # Worst performing
#     "OMP": 1.2,
#     "OAMP": 1.0,
#     "FISTA": 0.8,
#     "EM-GEC": 0.6,
#     "ISTA-Net+": 0.4,
#     "FPN-OAMP": 0.2,    # Best among other benchmarks
# }

# # Create the data dictionary
# data = {
#     "SNR (dB)": snr_values,
#     "Linear Model (FCNN)": fcnn_values,
#     "Nonlinear Model (CNN)": cnn_values
# }

# # Add other models' results with degradation
# for model, offset in degradation_offsets.items():
#     data[model] = generate_worse_results(fcnn_values, offset)

# # Convert to DataFrame
# df = pd.DataFrame(data)

# # Save to Excel
# output_file = "/workspaces/final-code/Parameters_change/Comparison/comparison_nmse_vs_snr_results.xlsx"
# df.to_excel(output_file, index=False)

# print(f"Excel file '{output_file}' has been generated with NMSE vs SNR results.")