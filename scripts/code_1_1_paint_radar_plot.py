import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

# --- 1. 从Excel加载数据 ---
def load_data_from_excel(file_path, sheet_name):
    """
    从指定的Excel文件和工作表中加载数据。

    :param file_path: Excel文件的完整路径。
    :param sheet_name: 要读取的工作表的名称。
    :return: 一个Pandas DataFrame，如果加载失败则返回None。
    """
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        print(f"成功从 '{file_path}' 的 '{sheet_name}' 工作表加载数据。")
        print("数据预览:")
        print(df.head())
        return df
    except FileNotFoundError:
        print(f"错误：文件未找到，请检查路径是否正确: {file_path}")
        return None
    except Exception as e:
        # 捕获其他可能的错误，例如工作表不存在
        print(f"加载Excel文件时发生错误: {e}")
        print("请确保文件名、工作表名和列名都正确无误。")
        return None

# --- 2. 绘图函数 (与之前相同，无需修改) ---

def plot_radar_chart(df):
    """
    绘制雷达图，比较不同网络的性能指标。
    绘制线条，不含填充，并包含图例。
    """
    # 选择要比较的指标
    metrics = ['Prec', 'Recall', 'F1']
    
    # 选择要比较的网络 (例如：基线、SOTA和我们的模型)
    models_to_compare = ['UNet', 'CDNet', 'FC-conc', 'FC-diff', 'SNUNet', 'P2V', 'LUNet', 'BIT', 'SPADANet']
    
    # 筛选出要绘图的数据，并检查模型是否存在
    data_to_plot = df[df['Network Name'].isin(models_to_compare)].set_index('Network Name').reindex(models_to_compare)
    if data_to_plot.isnull().values.any():
        print(f"警告: 在数据中找不到以下部分或全部模型，无法绘制雷达图: {models_to_compare}")
        return
        
    # 雷达图的设置
    labels = np.array(metrics)
    n_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    for model_name, row in data_to_plot.iterrows():
        stats = row[metrics].values.tolist()
        stats += stats[:1]
        ax.plot(angles, stats, label=model_name, linewidth=2.5, linestyle='solid')

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=14)
    ax.set_title('不同网络模型性能对比雷达图', size=20, color='black', y=1.1)
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)

    print("\n正在显示雷达图预览（仅线条）...")
    plt.show()

    save_path = r'scripts\paint_perform_compare\radar_charts\network_radar_chart.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"雷达图已保存至: {save_path}")

    # --- 3. 主程序执行 ---
if __name__ == "__main__":
    # 设置matplotlib以支持中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # ==================== 用户配置区域 ====================
    # 请在这里修改您的Excel文件路径和工作表名称
    excel_file_path = r"D:\DLProjects\Project_1_SemiDA\半监督实验资料\半监督结果\测试半监督结果绘图.xlsx"
    sheet = "模型比较"
    # ====================================================

    # 加载数据
    main_df = load_data_from_excel(excel_file_path, sheet)

    # 如果数据加载成功，则执行绘图
    if main_df is not None:
        # 检查必要的列是否存在
        required_columns = ['Network Name', 'Year', 'Prec', 'Recall', 'F1', 'Kappa', 'Params']
        if not all(col in main_df.columns for col in required_columns):
            print("错误：Excel文件中缺少必要的列。请确保包含以下所有列：")
            print(required_columns)
            sys.exit() # 退出程序

        # 调用绘图函数
        plot_radar_chart(main_df.copy())