import matplotlib.pyplot as plt
import numpy as np

# Data for Stridor and Control
categories = ['FIMO', 'Reg', 'RP', 'Deep']
stridor_values = [72, 70, 70, 52]
control_values = [42, 41, 43, 30]

# Total values for note
total_stridor = sum(stridor_values)
total_control = sum(control_values)

# Create a bar chart for Stridor and Control
x = np.arange(len(categories))  # label locations
width = 0.35  # width of the bars

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, stridor_values, width, label='Stridor', color='blue', alpha=0.6)
bars2 = ax.bar(x + width/2, control_values, width, label='Control', color='orange', alpha=0.6)

# Add some text for labels, title, and custom x-axis tick labels, etc.
ax.set_xlabel('Categories')
ax.set_ylabel('Values')
ax.set_title('Values for Stridor and Control')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

# Adding the values on top of the bars
for bar in bars1 + bars2:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, int(yval), ha='center', va='bottom')

# Adding a note of the total numbers
total_count = f"Total Stridor: {total_stridor}, Total Control: {total_control}"
plt.figtext(0.99, 0.01, total_count, horizontalalignment='right')

plt.show()
