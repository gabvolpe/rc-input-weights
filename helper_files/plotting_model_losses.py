import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

# === Configuration ===# Set seaborn style
sns.set(style="whitegrid")

# Dynamic Excel path
excel_path = os.path.join(os.getcwd(), 'Losses_Read-in_FixedRL.xlsx')

# Sheet names
sheet_name_median = 'Exp1_Sine Wave_Median'
sheet_name_iqr = 'Exp1_Sine Wave_IQR'
cols = "A:D"         # Columns A, B, C, D
start_row = 0        # B2 = row 0 → starts at row 1
# Number of rows in the sheet
df_median = pd.read_excel(excel_path, sheet_name=sheet_name_median)
n_rows = df_median.shape[0]      
print(n_rows)   

# === Read the data ===
df = pd.read_excel(
    excel_path,
    sheet_name=sheet_name_median,
    usecols=cols,
    skiprows=start_row,
    nrows=n_rows,
    header=None  # if there are no column headers in Excel
)

# Optional: name the columns for clarity
df.columns = ["Model A (Baseline)", "Model B", "Model C", "Model D"]

# === Plot boxplots for each model(with quantiles)===
import numpy as np

# Prepare data for boxplot
data = [df[col].dropna() for col in df.columns]

# Plot
plt.figure(figsize=(8, 6))
box = plt.boxplot(data, labels=df.columns)

# Calculate and annotate quartiles
for i, column_data in enumerate(data, start=1):
    q1 = np.percentile(column_data, 25)
    median = np.percentile(column_data, 50)
    q3 = np.percentile(column_data, 75)
    iqr = q3 - q1

    # Annotate values (adjust y positions slightly to avoid overlap)
    plt.text(i, q1 - 0.0003, f"Q1: {q1:.4f}", ha='center', va='top', fontsize=8) # q1 - 0.0004 to adjust according to graph
    plt.text(i + 0.5, median + 0.0001, f"Median: {median:.4f}", ha='center', va='bottom', fontsize=8) # i +0.5 to make it write next to the box, not inside, otherwise just i; median + 0.0001 to make it slight above the orange line (if i, instead of i+ 0.5)
    plt.text(i, q3 + 0.0004, f"Q3: {q3:.4f}", ha='center', va='bottom', fontsize=8)
    
    # Annotate IQR slightly higher than the upper top (0.001 or 0.002)
    plt.text(i, np.max(column_data) + 0.0004, f"IQR: {iqr:.4f}", ha='center', va='bottom', fontsize=8, color='purple')

plt.title("Exp1: Error distributions (Sine Wave)")
plt.ylabel("Loss (MSE)")
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

'''
# === Plot boxplots for each model(without quantiles)===
plt.figure(figsize=(8, 6))
plt.boxplot([df[col].dropna() for col in df.columns],
            labels=df.columns,
            patch_artist=True)

plt.title("Experiment 1: Error distributions (Sine Wave)")
plt.ylabel("Loss (MSE)")
plt.grid(True)
plt.tight_layout()
plt.show()
'''
'''
# Boxplots with colors (without Quantiles)
import matplotlib.pyplot as plt

# Define custom colors for each model
box_colors = ['skyblue', 'lightgreen', 'lightcoral', 'plum']

# Create the plot
plt.figure(figsize=(8, 6))
box = plt.boxplot([df[col].dropna() for col in df.columns],
                  labels=df.columns,
                  patch_artist=True)

# Color the boxes
for patch, color in zip(box['boxes'], box_colors):
    patch.set_facecolor(color)

plt.title("Loss Distributions per Model")
plt.ylabel("Loss (MSE)")
plt.grid(True)
plt.tight_layout()
plt.show()
'''

'''
Print Quantiles and Deviations'''
for model in df.columns:
    losses = df[model].dropna()

    q1 = losses.quantile(0.25)
    q2 = losses.quantile(0.50)  # median
    q3 = losses.quantile(0.75)

    iqr = q3 - q1  # Interquartile Range
    lower_dev = q2 - q1
    upper_dev = q3 - q2

    print(f"Model: {model}")
    print(f"  Q1: {q1:.6f}") # 6 digits after the decimal
    print(f"  Median (Q2): {q2:.6f}")
    print(f"  Q3: {q3:.6f}")
    print(f"  IQR: {iqr:.6f}")
    print(f"  Deviation below median: {lower_dev:.6f}")
    print(f"  Deviation above median: {upper_dev:.6f}")
    print()
