import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from adjustText import adjust_text

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
    

def plot_scatter_chart_performance_vs_params(df, annotation_fontsize=9, highlight_fontsize=11):
    """
    Draws a scatter plot comparing performance (Kappa) vs. model parameters (Params).
    - Font is set to Arial globally.
    - All labels are in English.
    - Annotations are to the right of the points.
    - Annotation font size is configurable.
    """
    fig, ax = plt.subplots(figsize=(10, 7))


    # 用于收集所有文本标签对象的列表
    texts = []

    for i, row in df.iterrows():
        x, y = row['Params'], row['Kappa']
        name = row['Network Name']
        
        ax.scatter(x, y, s=60, c='#1f77b4', alpha=0.6, zorder=2)
        
        # 创建文本对象并添加到列表中
        text_properties = {
            'x': x,
            'y': y,
            's': name,
            'fontsize': annotation_fontsize,
            'color': 'black',
            'fontweight': 'normal'
        }
        texts.append(ax.text(**text_properties))

        # # Highlight your model (SPADANet)
        # if name == 'SPADANet':
        #     ax.scatter(x, y, s=150, c='red', alpha=0.8, edgecolors='black', zorder=3, label='SPADANet (Our Model)')
        #     # Place text to the right, slightly larger and bold
        #     ax.text(x + text_offset_x, y, name, fontsize=highlight_fontsize, color='red', fontweight='bold', verticalalignment='center')
        # else:
        #     # Plot other models
        #     ax.scatter(x, y, s=80, c='#1f77b4', alpha=0.6, zorder=2) # Using a standard blue color
        #     # Place text to the right
        #     ax.text(x + text_offset_x, y, name, fontsize=annotation_fontsize, verticalalignment='center')

    # --- 核心步骤：使用 adjust_text 智能调整标签位置 ---
    # adjust_text 会移动标签以避免重叠，并可以绘制线条连接标签和原始点
    adjust_text(texts, 
                ax=ax, 
                force_static=(1.0, 1.0),
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.7)
                )
    
    # --- English Labels and Title ---
    ax.set_xlabel('Model Parameters (Millions)', fontsize=14)
    ax.set_ylabel('Kappa Coefficient', fontsize=14)
    ax.set_title('Performance vs. Model Parameters', fontsize=18)
    
    ax.grid(True, linestyle='--', alpha=0.6)
    # ax.legend(fontsize=12)

    print("\nDisplaying preview for Performance vs. Parameters scatter plot...")
    plt.show()

    # Save the figure
    save_path = r'scripts\paint_perform_compare\scatter_plots\performance_vs_params_scatter_english.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Scatter plot saved to: {save_path}")


def plot_scatter_chart_performance_vs_year(df, annotation_fontsize=9, highlight_fontsize=11):
    """
    绘制性能vs年份散点图，具有以下特性：
    - X轴显示所有年份。
    - 使用adjustText智能放置标签，避免重叠。
    - 所有标签和标题为英文，字体为Arial。
    """
    fig, ax = plt.subplots(figsize=(12, 8)) # 稍微增大画布尺寸以容纳标签
    
    # 用于收集所有文本标签对象的列表
    texts = []

    # 遍历数据点，创建散点和文本对象
    for i, row in df.iterrows():
        # 这里不再添加随机抖动，因为adjustText会处理位置
        x, y = row['Year'], row['Kappa']
        name = row['Network Name']
        
        ax.scatter(x, y, s=60, c='#1f77b4', alpha=0.6, zorder=2)
        # # 绘制散点
        # if name == 'SPADANet':
        #     ax.scatter(x, y, s=150, c='red', alpha=0.8, edgecolors='black', zorder=3, label='SPADANet (Our Model)')
        # else:
        #     ax.scatter(x, y, s=80, c='#1f77b4', alpha=0.6, zorder=2)
            
        # 创建文本对象并添加到列表中
        text_properties = {
            'x': x,
            'y': y,
            's': name,
            'fontsize': annotation_fontsize,
            'color': 'black',
            'fontweight': 'normal'
        }
        texts.append(ax.text(**text_properties))

    # --- 核心步骤：使用 adjust_text 智能调整标签位置 ---
    # adjust_text 会移动标签以避免重叠，并可以绘制线条连接标签和原始点
    adjust_text(texts, 
                ax=ax, 
                force_text=(99.0, 99.0),
                force_static=(2.0, 2.0),
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.7)
                )

    # --- 英文标签和标题 ---
    ax.set_xlabel('Publication Year', fontsize=14)
    ax.set_ylabel('F1-Score', fontsize=14)
    ax.set_title('Performance vs. Publication Year', fontsize=18)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # --- 需求 1：在X轴上显示所有年份 ---
    # 1. 获取数据中所有唯一且排序后的年份
    unique_years = sorted(df['Year'].unique())
    # 2. 将X轴的刻度设置为这些年份
    ax.set_xticks(unique_years)
    
    # 旋转刻度标签以防拥挤
    plt.xticks(rotation=45)
    
    # 调整Y轴范围，为标签留出更多空间
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max)
    
    # ax.legend(fontsize=12)

    print("\nDisplaying preview for smart Performance vs. Year scatter plot...")
    plt.show()

    save_path = 'performance_vs_year_scatter_english.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Smart scatter plot saved to: {save_path}")


# --- 3. 主程序执行 ---
if __name__ == "__main__":
    # 设置matplotlib以支持中文显示
    plt.rcParams['font.sans-serif'] = ['Arial']  # 黑体
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
        plot_scatter_chart_performance_vs_params(main_df.copy(), annotation_fontsize=16, highlight_fontsize=18)
        plot_scatter_chart_performance_vs_year(main_df.copy(), annotation_fontsize=16, highlight_fontsize=18)