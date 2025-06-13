import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from adjustText import adjust_text

# --- 1. DATA LOADING (as before) ---
def load_data_from_excel(file_path, sheet_name):
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        print(f"Successfully loaded data from '{file_path}' (Sheet: '{sheet_name}').")
        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# --- 2. CONFIGURATION: GROUPING AND STYLES ---
group_mapping = {
    'STANet': 'Self Attention based', 
    'HANet': 'Self Attention based', 
    'DSIFN': 'FCN based', 
    'SNUNet': 'Self Attention based',
    'BIT': 'Self Attention based', 
    'DTCDSCN': 'Feature Fusion', 
    'ChangeFormer': 'Self Attention based',
    'CGNet': 'U-Net based', 
    'TTP': 'Foundation Model based',
    'Changer': 'Feature Fusion',
    'BAN': 'Foundation Model based',
    'ChangeLN': 'Feature Fusion',
    'HeteCD': 'Self Attention based'
}
color_palette = {
    'U-Net based': '#1f77b4', 
    'FCN based': '#ff7f0e', 
    'Self Attention based': '#2ca02c',
    'Feature Fusion': "#de54a9",
    'Foundation Model based': "#96a02c"
}
marker_palette = {
    'U-Net based': 'o', 
    'FCN based': 's', 
    'Self Attention based': '^',
    'Feature Fusion': 'h',
    'Foundation Model based': 'd'
}

# --- 3. UNIFIED PLOTTING FUNCTION (from above) ---
def plot_grouped_scatter(df, x_col, y_col, title, xlabel, ylabel, filename, is_year_plot=False, annotation_fontsize=16):
    fig, ax = plt.subplots(figsize=(12, 8))
    texts = []
    categories = sorted(df['Category'].unique(), key=list(color_palette.keys()).index)
    
    for category in categories:
        group_df = df[df['Category'] == category]
        if group_df.empty: continue
        color, marker = color_palette[category], marker_palette[category]
        ax.scatter(group_df[x_col], 
                   group_df[y_col], 
                   s=80,
                   c=color, 
                   marker=marker, 
                   label=category, 
                   alpha=0.8, 
                   edgecolors='black',
                   linewidth=0.5,
                   zorder=3 if marker == '*' else 2)
        for i, row in group_df.iterrows():
            texts.append(ax.text(row[x_col], 
                                 row[y_col], 
                                 row['Network Name'],
                                 fontsize=annotation_fontsize,
                                 color='black',
                                 fontweight='normal'))

    adjust_text(texts, 
                ax=ax, 
                force_text=(99.0, 99.0),
                force_static=(2.0, 2.0),
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.7))

    # ax.set_title(title, fontsize=18); 
    ax.set_xlabel(xlabel, fontsize=14); ax.set_ylabel(ylabel, fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)
    if is_year_plot:
        ax.set_xticks(sorted(df['Year'].unique())); plt.xticks(rotation=0)
    ax.legend(title='Model Category', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11, title_fontsize=12)
    print(f"\nDisplaying preview for '{title}'..."); plt.show()
    fig.savefig(filename, dpi=300, bbox_inches='tight'); print(f"Plot saved to: {filename}")


# --- 4. MAIN EXECUTION SCRIPT ---
if __name__ == "__main__":
    try:
        plt.rcParams['font.family'] = 'Arial'
        print("Font set to Arial.")
    except:
        plt.rcParams['font.family'] = 'sans-serif'
        print("Arial font not found. Using default sans-serif font.")
    
    excel_file_path = r"D:\DLProjects\Project_1_SemiDA\半监督实验资料\半监督结果\测试半监督结果绘图.xlsx"
    sheet = "HETECD模型比较"
    
    main_df = load_data_from_excel(excel_file_path, sheet)

    if main_df is not None:
        # Add the 'Category' column based on the mapping
        main_df['Category'] = main_df['Network Name'].map(group_mapping)
        
        # Check if any network was not mapped
        if main_df['Category'].isnull().any():
            unmapped = main_df[main_df['Category'].isnull()]['Network Name'].tolist()
            print(f"WARNING: The following networks were not found in the group_mapping: {unmapped}")
        
        # --- Generate the plots using the new unified function ---
        plot_grouped_scatter(df=main_df, x_col='Params', y_col='F1',
                             title='Performance vs. Model Parameters by Architecture',
                             xlabel='Model Parameters (M)', ylabel='F1-Score',
                             filename='scripts\paint_perform_compare\scatter_plots\grouped_perf_vs_params_isprs.png')
                             
        plot_grouped_scatter(df=main_df, x_col='Year', y_col='F1',
                             title='Performance vs. Publication Year by Architecture',
                             xlabel='Publication Year', ylabel='F1-Score',
                             filename='scripts\paint_perform_compare\scatter_plots\grouped_perf_vs_year_isprs.png',
                             is_year_plot=True)
        
        print("\nAll grouped plots have been generated and saved.")
    else:
        print("\nExecution stopped due to data loading failure.")