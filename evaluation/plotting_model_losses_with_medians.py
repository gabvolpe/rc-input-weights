'''
Systems: 'Sine Wave', 'Mackey-Glass', 'Lorenz', 'NARMA-10'
Tasks: 'Sine-to-Cosine$^2$', 'Mackey-Glass', 'Lorenz', 'NARMA-10'
Constraint Sets: '1', '2', '3'
'''

system ="Sine Wave"
task="Sine-to-Cosine$^2$"
constraint_set= "1"

# --- MODEL LOSSES WITH MEDIANS AND IQR ---
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# === Configuration ===
sns.set(style="whitegrid")
plt.rcParams.update({'figure.autolayout': True})

# Excel path and sheet names
excel_path = os.path.join(os.getcwd(), 'Losses_Read-in_FixedRL.xlsx')
sheet_name_median = 'Exp'+constraint_set+'_'+ system+'_Median'
cols = "A:I"
start_row = 1

# === Load data ===
df_median = pd.read_excel(excel_path, sheet_name=sheet_name_median)
n_rows = df_median.shape[0]

df_header_row = pd.read_excel(excel_path, sheet_name=sheet_name_median, nrows=1, header=None).iloc[0]

sigma_cell = df_header_row.iloc[9]
sigma = str(sigma_cell).replace("sigma=", "").strip()

gaussian_col_name = None
for col_name in df_header_row:
    if f"Gaussian_sd_{sigma}" in str(col_name):
        gaussian_col_name = col_name
        break
if gaussian_col_name is None:
    raise ValueError(f"No column matching 'Gaussian_sd_{sigma}' found!")

df_full = pd.read_excel(excel_path, sheet_name=sheet_name_median,
                        usecols=cols, skiprows=start_row, nrows=n_rows, header=None)
df_full.columns = df_header_row[:df_full.shape[1]]

columns_to_keep = ["Uniform", gaussian_col_name, "Double-Gaussian", "Laplace", "Power-Law"]
df = df_full[columns_to_keep]
df.columns = ["Uniform (Baseline)", f"Gaussian sd={sigma}", f"Double Gauss σ={sigma}", "Laplace", "Power Law"]

colors = ['#0000CD', '#D55E00', '#009E73', '#DC143C', '#B8476D']
model_names = df.columns.tolist()
positions = np.arange(1, len(model_names)+1)

plt.figure(figsize=(8, 6), facecolor='white')

# Plot violin plots with quartile lines
for i, col in enumerate(df.columns):
    series = df[col].dropna()
    
    # Violin plot
    vp = plt.violinplot(series, positions=[positions[i]], showmedians=False, showextrema=False)
    
    # Customize appearance
    for body in vp['bodies']:
        body.set_facecolor(colors[i])
        body.set_alpha(0.7)
        body.set_edgecolor('black')
        body.set_linewidth(1.2)
    
    # Quartiles
    q1 = np.percentile(series, 25)
    median = np.percentile(series, 50)
    q3 = np.percentile(series, 75)
    iqr = q3 - q1
    
    # Draw quartile lines inside the violin
    plt.plot([positions[i]-0.2, positions[i]+0.2], [q1, q1], color='black', linewidth=1.5)  # Q1
    plt.plot([positions[i]-0.2, positions[i]+0.2], [median, median], color='white', linewidth=2) # Median
    plt.plot([positions[i]-0.2, positions[i]+0.2], [q3, q3], color='black', linewidth=1.5)  # Q3
    
    '''# Annotate median slightly above
    plt.text(positions[i], median + 0.00005, f"{median:.4f}",
             ha='center', va='bottom', fontsize=9, color='white', fontweight='bold')'''
    
    # Annotate IQR slightly above top
    max_value = series.max()
    plt.text(positions[i], max_value + 0.01*(max_value - series.min()),
             f"IQR: {iqr:.4f}", ha='center', va='bottom', fontsize=8, color='purple', fontweight='bold')

# Annotate x-axis labels in corresponding colors
plt.xticks(positions, model_names, fontsize=10, fontweight='bold', rotation=0)
ax = plt.gca()
for tick_label, color in zip(ax.get_xticklabels(), colors):
    tick_label.set_color(color)

# Titles and labels
#plt.title("Exp"+experiment+": Error Distributions ("+task+")", fontsize=12)
plt.ylabel("Loss (MSE)", fontsize=11)
plt.grid(False, axis='y')
plt.grid(False, axis='x')

plt.tight_layout()
plt.show()

