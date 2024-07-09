import matplotlib.pyplot as plt
import numpy as np

# Data for the categories
categories = ['Avid', 'iPad', '12inch', '1inch', 'Unknown']
FIMO = [53, 42, 16, 0, 3]
Reg = [53, 41, 14, 0, 3]
RP = [53, 41, 15, 1, 3]
Deep = [41, 41, 0, 0, 0]

# Data for the groups
groups = ['FIMO', 'Reg', 'RP', 'Deep']
data = [FIMO, Reg, RP, Deep]
data = np.array(data).T  # Transpose the data for stacked bar chart

# Color palette
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Create a stacked bar chart
fig, ax = plt.subplots(figsize=(10, 6))

bottom = np.zeros(len(groups))

for idx, category in enumerate(categories):
    bars = ax.bar(groups, data[idx], bottom=bottom, label=category, color=colors[idx], alpha=0.7)
    for bar in bars:
        height = bar.get_height()
        if height > 0:  # Only add text for non-zero segments
            ax.text(
                bar.get_x() + bar.get_width() / 2, 
                bar.get_y() + height / 2, 
                int(height), 
                ha='center', 
                va='center', 
                fontsize=10, 
                color='white'
            )
    bottom += data[idx]

# Add some text for labels, title, and custom x-axis tick labels, etc.
ax.set_xlabel('Groups')
ax.set_ylabel('Values')
ax.set_title('Distribution of Categories Across Groups')
ax.legend(loc='upper right')

# Adding total counts at the bottom right
total_counts = f"Total FIMO: {sum(FIMO)}, Total Reg: {sum(Reg)}, Total RP: {sum(RP)}, Total Deep: {sum(Deep)}"
plt.figtext(0.99, 0.01, total_counts, horizontalalignment='right')

plt.show()
