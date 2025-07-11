import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import numpy as np

# Set seaborn style
sns.set(style="whitegrid")

# Dynamic Excel path
excel_path = os.path.join(os.getcwd(), 'Losses_Read-in_FixedRL.xlsx')

# Sheet names
sheet_name_median = 'Exp2_Mackey-Glass_Median'
sheet_name_iqr = 'Exp2_Mackey-Glass_IQR'

# Read IQR sheet
df_iqr = pd.read_excel(excel_path, sheet_name=sheet_name_iqr)

# Model names (correspond to columns A-D)
model_names = ['Model A (Baseline)', 'Model B', 'Model C', 'Model D']
colors = ['#1C3F60','#A67C52','#3E885B','#B8476D'] # midnight blue, walnut bronze, botanical green, cranberry rose, or ['blue', 'orange', 'red', 'purple']

# Plot setup
plt.figure(figsize=(10, 6))

# Add vertical light green line at x=0
plt.axvline(x=0.0, color='mediumseagreen', linestyle='--', linewidth=2, label='Null Hypothesis(H0)')

# Histogram bins setup (shared bins for all models)
bins = 30
all_data = pd.concat([df_iqr.iloc[:, i].dropna() for i in range(len(model_names))])
bin_edges = np.linspace(all_data.min(), all_data.max(), bins + 1)
bin_width = bin_edges[1] - bin_edges[0]

# Plot histogram + manual KDE for each model
for i, model in enumerate(model_names):
    data = df_iqr.iloc[:, i].dropna()
    
    # Histogram: frequency counts, fixed bins, no KDE here
    plt.hist(data, bins=bin_edges, histtype='step', linewidth=1.0,
             label=model, color=colors[i], fill=False, zorder=2)

    # Compute KDE using scipy
    kde = gaussian_kde(data)
    x_vals = np.linspace(bin_edges[0], bin_edges[-1], 500)
    kde_vals = kde(x_vals)
    
    # Scale KDE to match histogram counts: counts = density * (number of points) * bin width
    scaled_kde = kde_vals * len(data) * bin_width
    
    # Plot KDE curve manually
    plt.plot(x_vals, scaled_kde, color=colors[i], linewidth=2.0, alpha=0.9, zorder=3)

# Customize axes so they meet at x = -0.5 and y = 0
ax = plt.gca()
ax.set_xlim(left=-0.001)

# Move left spine (y-axis) to x = -0.5
ax.spines['left'].set_position(('data', -0.001))
ax.spines['left'].set_visible(True)

# Move bottom spine (x-axis) to y = 0
ax.spines['bottom'].set_position(('data', 0))
ax.spines['bottom'].set_visible(True)

# Hide top and right spines for cleaner look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Set ticks position
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

# Title and axis labels
plt.title('Exp1 - IQR Frequency for System')
plt.xlabel('Interquartile Range (IQR)')
plt.ylabel('Frequency')

plt.legend()
plt.tight_layout()
plt.show()
