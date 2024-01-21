import argparse
import pandas as pd
import matplotlib.pyplot as plt

# python3 plot_csv.py matrix_torch.csv matrix_torch.png GFLOPS_cpu GFLOPS_cuda 

def plot_from_csv(input_file, output_file, columns):
    # 读取 CSV 文件
    df = pd.read_csv(input_file)

    # 设置第一列为横轴
    x = df.iloc[:, 0]

    # 绘制指定的列
    plt.figure()
    for col in columns:
        if col in df.columns:
            plt.plot(x, df[col], 'o-', label=col)
    plt.xlabel('Matrix Size (N)')
    plt.ylabel('Performance (GFLOPS)')
    plt.title('Matrix Multiplication Performance')
    plt.legend()

    # 保存图表到文件
    plt.savefig(output_file)
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='python3 plot_csv.py matrix_torch.csv matrix_torch.png GFLOPS_cpu GFLOPS_cuda')
    parser.add_argument('input_file', type=str, help='Input CSV file')
    parser.add_argument('output_file', type=str, help='Output image file')
    parser.add_argument('columns', nargs='+', help='Columns to plot')
    args = parser.parse_args()

    plot_from_csv(args.input_file, args.output_file, args.columns)
