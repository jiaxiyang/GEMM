import sys
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
# df = pd.DataFrame(data)
# df = pd.read_csv('performance_data.csv')
df = pd.read_csv(sys.argv[1])

# Set the size for our plot
plt.figure(figsize=(15, 7))

n_groups = len(df)
n_cols = len(df.columns[1:])
bar_width = 0.85 / n_cols  # 调整分母以改变条形宽度

# 创建一个索引数组，确保条形在组中居中显示
index = np.arange(n_groups) + 0.5 * (1 - n_cols * bar_width)

# Create an index for the groups
# index = np.arange(n_groups)

# Plotting each version's performance with an offset for each bar
for i, column in enumerate(df.columns[1:]):  # Assuming the first column is for matrix sizes
    bars = plt.bar(index + i * bar_width, df[column], bar_width, label=column)
    # Adding the text labels above the bars
    # for bar in bars:
        # yval = bar.get_height()
        # plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), va='bottom', ha='center')

# Adding labels and title
plt.xlabel('Matrix Size (M=N=K)')
plt.ylabel('GFLOPS')
plt.title('My GEMM Performance on RTX 3080 Ti')

# 添加图例并将其放置在图表外部
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

# 设置x轴刻度的位置
plt.xticks(index + (n_cols - 1) * bar_width / 2, df['M'])

# Display the plot with a tight layout
plt.tight_layout()

# Save the plot
plt.savefig(sys.argv[2])
# plt.show()
