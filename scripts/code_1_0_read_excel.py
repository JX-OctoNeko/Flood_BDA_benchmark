import pandas as pd
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

if __name__ == "__main__":
    # 读取Excel文件
    excel_file_path = r"D:\DLProjects\Project_1_SemiDA\半监督实验资料\半监督结果\测试半监督结果绘图.xlsx"
    sheet = "模型比较"

    # 加载数据
    main_df = load_data_from_excel(excel_file_path, sheet)

    if main_df is not None:
        # 检查必要的列是否存在
        required_columns = ['Network Name', "Year", "Prec", "Recall", "F1", "Kappa", "Params"]
        if not all(col in main_df.columns for col in required_columns):
            print("ERROR: Excel require all the columns:")
            print(required_columns)
            sys.exit()