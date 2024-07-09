import matplotlib.pyplot as plt

# Data for Stridor and Control
categories = ['Stridor', 'Control']
values = [41, 17]

# Create a bar chart
plt.figure(figsize=(10, 6))
bars = plt.bar(categories, values, color=['blue', 'orange'])
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Values for Stridor and Control')

# Adding the values on top of the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, int(yval), ha='center', va='bottom')

# Adding a note of the total numbers
plt.figtext(0.99, 0.01, f'Total: {sum(values)}', horizontalalignment='right')

plt.show()
