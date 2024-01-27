# import pandas as pd
# import matplotlib.pyplot as plt

# # Load the data from CSV
# df = pd.read_csv('test.csv')

# # Set the size for our plot
# plt.figure(figsize=(15, 7))

# # Plotting each version's performance
# for column in df.columns[1:]:  # Assuming the first column is for matrix sizes
#     plt.bar(df.index, df[column], label=column)

# # Adding labels and title
# plt.xlabel('Matrix Size (M=N=K)')
# plt.ylabel('GFLOPS')
# plt.title('SGEMM Performance')

# # Adding a legend
# plt.legend()

# plt.savefig('result.png')
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# # Load the data from CSV
# # Replace 'your_csv_file.csv' with the path to your actual CSV file
# # df = pd.read_csv('test.csv')
# # Create a sample CSV data
# data = {
#     'Matrix Size': [128, 256, 512, 1024, 2048],
#     'naive': [144, 243, 259, 235, 212],
#     'V1': [139, 691, 1912, 2180, 2610],
#     'V2': [163, 847, 2246, 2628, 2765],
#     'V3': [240, 1145, 2925, 3396, 3478],
#     'cublas': [409, 1201, 2155, 2503, 3682]
# }

# # Convert the dictionary to a DataFrame
# df = pd.DataFrame(data)
# # Set the size for our plot
# plt.figure(figsize=(15, 7))

# # Number of groups
# n_groups = len(df)

# # Create a bar width
# bar_width = 0.15

# # Create an index for the groups
# index = np.arange(n_groups)

# # Plotting each version's performance with an offset for each bar
# for i, column in enumerate(df.columns[1:]):  # Assuming the first column is for matrix sizes
#     plt.bar(index + i*bar_width, df[column], bar_width, label=column)

# # Adding labels and title
# plt.xlabel('Matrix Size (M=N=K)')
# plt.ylabel('GFLOPS')
# plt.title('SGEMM Performance')

# # Adding the legend outside of the plot
# plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")

# # Set the position of the x ticks
# plt.xticks(index + bar_width, df['Matrix Size'])

# # Display the plot with a tight layout
# plt.tight_layout()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Sample data
data = {
    'Matrix Size': [128, 256, 512, 1024, 2048],
    'naive': [144, 243, 259, 235, 212],
    'V1': [139, 691, 1912, 2180, 2610],
    'V2': [163, 847, 2246, 2628, 2765],
    'V3': [240, 1145, 2925, 3396, 3478],
    'cublas': [409, 1201, 2155, 2503, 3682]
}

# Convert the dictionary to a DataFrame
df = pd.DataFrame(data)

# Set the size for our plot
plt.figure(figsize=(15, 7))

# Number of groups
n_groups = len(df)

# Create a bar width
bar_width = 0.15

# Create an index for the groups
index = np.arange(n_groups)

# Plotting each version's performance with an offset for each bar
for i, column in enumerate(df.columns[1:]):  # Assuming the first column is for matrix sizes
    bars = plt.bar(index + i * bar_width, df[column], bar_width, label=column)
    # Adding the text labels above the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), va='bottom', ha='center')

# Adding labels and title
plt.xlabel('Matrix Size (M=N=K)')
plt.ylabel('GFLOPS')
plt.title('SGEMM Performance')

# Adding the legend outside of the plot
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")

# Set the position of the x ticks
plt.xticks(index + bar_width, df['Matrix Size'])

# Display the plot with a tight layout
plt.tight_layout()

# Save the plot

plt.savefig('result.png')
# plt.show()
