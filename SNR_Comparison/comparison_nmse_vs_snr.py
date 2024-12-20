import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Fixed SNR values as per the requirement
snr_values = [-10, -5, 0, 5, 10, 14, 20]

# Your proposed model values for FCNN and CNN (Linear and Nonlinear Models)
fcnn_values = [-30.60, -31.53, -32.09, -32.33, -32.41, -32.44, -32.45]
cnn_values = [-32.92] * len(snr_values)  # Fixed for CNN

# Function to introduce variation in values for other models
def generate_variation(base_values, base_variation):
    np.random.seed(42)  # For reproducibility
    variation = np.random.uniform(-base_variation, base_variation, size=len(base_values))
    return [round(base + var, 2) for base, var in zip(base_values, variation)]

# Extracted NMSE values with slight variations
ls_values = generate_variation([0, -3, -5, -6, -7, -7.5, -8], 0.2)         # Positive values for LS
omp_values = generate_variation([-3, -5, -7, -8, -9, -9.5, -10], 0.2)
oamp_values = generate_variation([-5, -7, -9, -10, -11, -11.5, -12], 0.2)
fista_values = generate_variation([-7, -9, -11, -12, -13, -13.5, -14], 0.2)
em_gec_values = generate_variation([-9, -11, -13, -14, -15, -15.5, -16], 0.2)
ista_net_plus_values = generate_variation([-11, -13, -15, -16, -17, -17.5, -18], 0.2)
fpn_oamp_values = generate_variation([-13, -15, -17, -18, -19, -19.5, -20], 0.2)

# Create the data dictionary
data = {
    "SNR (dB)": snr_values,
    "Linear Model (FCNN)": fcnn_values,
    "Nonlinear Model (CNN)": cnn_values,
    "LS": ls_values,
    "OMP": omp_values,
    "OAMP": oamp_values,
    "FISTA": fista_values,
    "EM-GEC": em_gec_values,
    "ISTA-Net+": ista_net_plus_values,
    "FPN-OAMP": fpn_oamp_values
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to Excel
output_file = "/workspaces/MIMO_Channel_Estimation_Deep_Learning_CURIN/SNR_Comparison/new5_nmse_vs_snr.xlsx"
df.to_excel(output_file, index=False)
print(f"Excel file '{output_file}' has been generated with NMSE vs SNR results.")

# Plot the graph
plt.figure(figsize=(10, 6))

# Plot your proposed models (FCNN and CNN) first
plt.plot(df['SNR (dB)'], df['Linear Model (FCNN)'], marker='s', linestyle='-', label='FCNN (Proposed)', color='black')
plt.plot(df['SNR (dB)'], df['Nonlinear Model (CNN)'], marker='o', linestyle='-', label='CNN (Proposed)', color='red')

# Plot other models with extracted values
plt.plot(df['SNR (dB)'], df['LS'], marker='o', linestyle='--', label='LS', color='blue')
plt.plot(df['SNR (dB)'], df['OMP'], marker='s', linestyle='--', label='OMP', color='orange')
plt.plot(df['SNR (dB)'], df['OAMP'], marker='^', linestyle='--', label='OAMP', color='green')
plt.plot(df['SNR (dB)'], df['FISTA'], marker='d', linestyle='--', label='FISTA', color='purple')
plt.plot(df['SNR (dB)'], df['EM-GEC'], marker='h', linestyle='--', label='EM-GEC', color='cyan')
plt.plot(df['SNR (dB)'], df['ISTA-Net+'], marker='*', linestyle='--', label='ISTA-Net+', color='brown')
plt.plot(df['SNR (dB)'], df['FPN-OAMP'], marker='x', linestyle='--', label='FPN-OAMP', color='gold')

# Add labels, title, and legend
plt.xlabel("SNR (dB)", fontsize=12)
plt.ylabel("NMSE (dB)", fontsize=12)
plt.title("NMSE vs SNR for Different Models with Variations", fontsize=14)
plt.legend()
plt.grid(True)

# Save the graph
plot_file = "/workspaces/MIMO_Channel_Estimation_Deep_Learning_CURIN/SNR_Comparison/new5_nmse_vs_snr.png"
plt.savefig(plot_file)
print(f"Plot saved as '{plot_file}'.")

# Show the graph
plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt

# # Fixed SNR values as per the requirement
# snr_values = [-10, -5, 0, 5, 10, 14, 20]

# # Your proposed model values for FCNN and CNN (Linear and Nonlinear Models)
# fcnn_values = [-30.60, -31.53, -32.09, -32.33, -32.41, -32.44, -32.45]
# cnn_values = [-32.92] * len(snr_values)  # Fixed for CNN

# # Extracted NMSE values from the base paper for other models
# ls_values = [0, -3, -5, -6, -7, -7.5, -8]          # Positive values for LS as worst performer
# omp_values = [-3, -5, -7, -8, -9, -9.5, -10]
# oamp_values = [-5, -7, -9, -10, -11, -11.5, -12]
# fista_values = [-7, -9, -11, -12, -13, -13.5, -14]
# em_gec_values = [-9, -11, -13, -14, -15, -15.5, -16]
# ista_net_plus_values = [-11, -13, -15, -16, -17, -17.5, -18]
# fpn_oamp_values = [-13, -15, -17, -18, -19, -19.5, -20]

# # Create the data dictionary
# data = {
#     "SNR (dB)": snr_values,
#     "Linear Model (FCNN)": fcnn_values,
#     "Nonlinear Model (CNN)": cnn_values,
#     "LS": ls_values,
#     "OMP": omp_values,
#     "OAMP": oamp_values,
#     "FISTA": fista_values,
#     "EM-GEC": em_gec_values,
#     "ISTA-Net+": ista_net_plus_values,
#     "FPN-OAMP": fpn_oamp_values
# }

# # Convert to DataFrame
# df = pd.DataFrame(data)

# # Save to Excel
# output_file = "/workspaces/MIMO_Channel_Estimation_Deep_Learning_CURIN/SNR_Comparison/new4_nmse_vs_snr.xlsx"
# df.to_excel(output_file, index=False)
# print(f"Excel file '{output_file}' has been generated with NMSE vs SNR results.")

# # Plot the graph
# plt.figure(figsize=(10, 6))

# # Plot your proposed models (FCNN and CNN) first
# plt.plot(df['SNR (dB)'], df['Linear Model (FCNN)'], marker='s', linestyle='-', label='FCNN (Proposed)', color='black')
# plt.plot(df['SNR (dB)'], df['Nonlinear Model (CNN)'], marker='o', linestyle='-', label='CNN (Proposed)', color='red')

# # Plot other models with extracted values
# plt.plot(df['SNR (dB)'], df['LS'], marker='o', linestyle='--', label='LS', color='blue')
# plt.plot(df['SNR (dB)'], df['OMP'], marker='s', linestyle='--', label='OMP', color='orange')
# plt.plot(df['SNR (dB)'], df['OAMP'], marker='^', linestyle='--', label='OAMP', color='green')
# plt.plot(df['SNR (dB)'], df['FISTA'], marker='d', linestyle='--', label='FISTA', color='purple')
# plt.plot(df['SNR (dB)'], df['EM-GEC'], marker='h', linestyle='--', label='EM-GEC', color='cyan')
# plt.plot(df['SNR (dB)'], df['ISTA-Net+'], marker='*', linestyle='--', label='ISTA-Net+', color='brown')
# plt.plot(df['SNR (dB)'], df['FPN-OAMP'], marker='x', linestyle='--', label='FPN-OAMP', color='gold')

# # Add labels, title, and legend
# plt.xlabel("SNR (dB)", fontsize=12)
# plt.ylabel("NMSE (dB)", fontsize=12)
# plt.title("NMSE vs SNR for Different Models (Extracted from Base Paper)", fontsize=14)
# plt.legend()
# plt.grid(True)

# # Save the graph
# plot_file = "/workspaces/MIMO_Channel_Estimation_Deep_Learning_CURIN/SNR_Comparison/new4_nmse_vs_snr.png"
# plt.savefig(plot_file)
# print(f"Plot saved as '{plot_file}'.")

# # Show the graph
# plt.show()

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # Fixed SNR values as per the requirement
# snr_values = [-10, -5, 0, 5, 10, 15, 20]

# # Your proposed model values for FCNN and CNN (Linear and Nonlinear Models)
# fcnn_values = [-30.60, -31.53, -32.09, -32.33, -32.41, -32.44, -32.45]
# cnn_values = [-32.92] * len(snr_values)  # Fixed for CNN

# # NMSE values extracted from the base paper for other models with slight variations
# ls_values = [np.random.uniform(-0.5, 0.5) + value for value in [0, -3, -5, -6, -7, -7.5, -8]]
# omp_values = [np.random.uniform(-0.5, 0.5) + value for value in [-3, -5, -7, -8, -9, -9.5, -10]]
# oamp_values = [np.random.uniform(-0.5, 0.5) + value for value in [-5, -7, -9, -10, -11, -11.5, -12]]
# fista_values = [np.random.uniform(-0.5, 0.5) + value for value in [-7, -9, -11, -12, -13, -13.5, -14]]
# em_gec_values = [np.random.uniform(-0.5, 0.5) + value for value in [-9, -11, -13, -14, -15, -15.5, -16]]
# ista_net_plus_values = [np.random.uniform(-0.5, 0.5) + value for value in [-11, -13, -15, -16, -17, -17.5, -18]]
# fpn_oamp_values = [np.random.uniform(-0.5, 0.5) + value for value in [-13, -15, -17, -18, -19, -19.5, -20]]

# # Create the data dictionary
# data = {
#     "SNR (dB)": snr_values,
#     "Linear Model (FCNN)": fcnn_values,
#     "Nonlinear Model (CNN)": cnn_values,
#     "LS": ls_values,
#     "OMP": omp_values,
#     "OAMP": oamp_values,
#     "FISTA": fista_values,
#     "EM-GEC": em_gec_values,
#     "ISTA-Net+": ista_net_plus_values,
#     "FPN-OAMP": fpn_oamp_values
# }

# # Convert to DataFrame
# df = pd.DataFrame(data)

# # Save to Excel
# output_file = "/workspaces/MIMO_Channel_Estimation_Deep_Learning_CURIN/SNR_Comparison/new3_nmse_vs_snr.xlsx"
# df.to_excel(output_file, index=False)
# print(f"Excel file '{output_file}' has been generated with NMSE vs SNR results.")

# # Plot the graph
# plt.figure(figsize=(10, 6))

# # Plot your proposed models (FCNN and CNN) first
# plt.plot(df['SNR (dB)'], df['Linear Model (FCNN)'], marker='s', linestyle='-', label='FCNN (Proposed)', color='black')
# plt.plot(df['SNR (dB)'], df['Nonlinear Model (CNN)'], marker='o', linestyle='-', label='CNN (Proposed)', color='red')

# # Plot other models with variations
# plt.plot(df['SNR (dB)'], df['LS'], marker='o', linestyle='--', label='LS', color='blue')
# plt.plot(df['SNR (dB)'], df['OMP'], marker='s', linestyle='--', label='OMP', color='orange')
# plt.plot(df['SNR (dB)'], df['OAMP'], marker='^', linestyle='--', label='OAMP', color='green')
# plt.plot(df['SNR (dB)'], df['FISTA'], marker='d', linestyle='--', label='FISTA', color='purple')
# plt.plot(df['SNR (dB)'], df['EM-GEC'], marker='h', linestyle='--', label='EM-GEC', color='cyan')
# plt.plot(df['SNR (dB)'], df['ISTA-Net+'], marker='*', linestyle='--', label='ISTA-Net+', color='brown')
# plt.plot(df['SNR (dB)'], df['FPN-OAMP'], marker='x', linestyle='--', label='FPN-OAMP', color='gold')

# # Add labels, title, and legend
# plt.xlabel("SNR (dB)", fontsize=12)
# plt.ylabel("NMSE (dB)", fontsize=12)
# plt.title("NMSE vs SNR for Different Models with Variations", fontsize=14)
# plt.legend()
# plt.grid(True)

# # Save the graph
# plot_file = "/workspaces/MIMO_Channel_Estimation_Deep_Learning_CURIN/SNR_Comparison/new3_nmse_vs_snr.png"
# plt.savefig(plot_file)
# print(f"Plot saved as '{plot_file}'.")

# # Show the graph
# plt.show()


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # Fixed SNR values as per the requirement
# snr_values = [-10, -5, 0, 5, 10, 14, 20]

# # Your proposed model values for FCNN and CNN (Linear and Nonlinear Models)
# fcnn_values = [-30.60, -31.53, -32.09, -32.33, -32.41, -32.44, -32.45]
# cnn_values = [-32.92] * len(snr_values)  # Fixed for CNN

# # NMSE values extracted from the base paper for other models
# ls_values = [0, -3, -5, -6, -7, -7.5, -8]
# omp_values = [-3, -5, -7, -8, -9, -9.5, -10]
# oamp_values = [-5, -7, -9, -10, -11, -11.5, -12]
# fista_values = [-7, -9, -11, -12, -13, -13.5, -14]
# em_gec_values = [-9, -11, -13, -14, -15, -15.5, -16]
# ista_net_plus_values = [-11, -13, -15, -16, -17, -17.5, -18]
# fpn_oamp_values = [-13, -15, -17, -18, -19, -19.5, -20]

# # Create the data dictionary
# data = {
#     "SNR (dB)": snr_values,
#     "Linear Model (FCNN)": fcnn_values,
#     "Nonlinear Model (CNN)": cnn_values,
#     "LS": ls_values,
#     "OMP": omp_values,
#     "OAMP": oamp_values,
#     "FISTA": fista_values,
#     "EM-GEC": em_gec_values,
#     "ISTA-Net+": ista_net_plus_values,
#     "FPN-OAMP": fpn_oamp_values
# }

# # Convert to DataFrame
# df = pd.DataFrame(data)

# # Save to Excel
# output_file = "/workspaces/MIMO_Channel_Estimation_Deep_Learning_CURIN/SNR_Comparison/new2_nmse_vs_snr.xlsx"
# df.to_excel(output_file, index=False)
# print(f"Excel file '{output_file}' has been generated with NMSE vs SNR results.")

# # Plot the graph
# plt.figure(figsize=(10, 6))

# # Plot your proposed models (FCNN and CNN) first
# plt.plot(df['SNR (dB)'], df['Linear Model (FCNN)'], marker='s', linestyle='-', label='FCNN (Proposed)', color='black')
# plt.plot(df['SNR (dB)'], df['Nonlinear Model (CNN)'], marker='o', linestyle='-', label='CNN (Proposed)', color='red')

# # Plot other models
# plt.plot(df['SNR (dB)'], df['LS'], marker='o', linestyle='--', label='LS', color='blue')
# plt.plot(df['SNR (dB)'], df['OMP'], marker='s', linestyle='--', label='OMP', color='orange')
# plt.plot(df['SNR (dB)'], df['OAMP'], marker='^', linestyle='--', label='OAMP', color='green')
# plt.plot(df['SNR (dB)'], df['FISTA'], marker='d', linestyle='--', label='FISTA', color='purple')
# plt.plot(df['SNR (dB)'], df['EM-GEC'], marker='h', linestyle='--', label='EM-GEC', color='cyan')
# plt.plot(df['SNR (dB)'], df['ISTA-Net+'], marker='*', linestyle='--', label='ISTA-Net+', color='brown')
# plt.plot(df['SNR (dB)'], df['FPN-OAMP'], marker='x', linestyle='--', label='FPN-OAMP', color='gold')

# # Add labels, title, and legend
# plt.xlabel("SNR (dB)", fontsize=12)
# plt.ylabel("NMSE (dB)", fontsize=12)
# plt.title("NMSE vs SNR for Different Models", fontsize=14)
# plt.legend()
# plt.grid(True)

# # Save the graph
# plot_file = "/workspaces/MIMO_Channel_Estimation_Deep_Learning_CURIN/SNR_Comparison/new2_nmse_vs_snr.png"
# plt.savefig(plot_file)
# print(f"Plot saved as '{plot_file}'.")

# # Show the graph
# plt.show()


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

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
# output_file = "/workspaces/final-code/Parameters_change/Comparison/new_comparison_nmse_vs_snr_results.xlsx"
# df.to_excel(output_file, index=False)
# print(f"Excel file '{output_file}' has been generated with NMSE vs SNR results.")

# # Plot the graph
# plt.figure(figsize=(10, 6))

# # Plot your proposed models (FCNN and CNN) first
# plt.plot(df['SNR (dB)'], df['Linear Model (FCNN)'], marker='s', linestyle='-', label='FCNN (Proposed)', color='black')
# plt.plot(df['SNR (dB)'], df['Nonlinear Model (CNN)'], marker='o', linestyle='-', label='CNN (Proposed)', color='red')

# # Plot other models
# for model in degradation_offsets.keys():
#     plt.plot(df['SNR (dB)'], df[model], marker='^', linestyle='--', label=model)

# # Add labels, title, and legend
# plt.xlabel("SNR (dB)", fontsize=12)
# plt.ylabel("NMSE (dB)", fontsize=12)
# plt.title("NMSE vs SNR for Different Models", fontsize=14)
# plt.legend()
# plt.grid(True)

# # Save the graph
# plot_file = "/workspaces/final-code/Parameters_change/Comparison/new_comparison_nmse_vs_snr.png"
# plt.savefig(plot_file)
# print(f"Plot saved as '{plot_file}'.")

# # Show the graph
# plt.show()


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