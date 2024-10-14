import os

root_folder = r'C:\Users\student\DLModels\cdlab\CDLab\src\data\xView\Process2'  # 文件夹路径


# 遍历根目录下的test、train、val文件夹
for phase_folder in ['test', 'train', 'val']:
    phase_folder_path = os.path.join(root_folder, phase_folder)

    # 遍历label文件夹
    label_folder_path = os.path.join(phase_folder_path, 'label')

    # 遍历label文件夹下的子文件夹
    for disaster_folder in os.listdir(label_folder_path):
        disaster_folder_path = os.path.join(label_folder_path, disaster_folder)

        # 重命名文件夹
        new_disaster_folder_name = f"{disaster_folder}_target"
        new_disaster_folder_path = os.path.join(label_folder_path, new_disaster_folder_name)
        os.rename(disaster_folder_path, new_disaster_folder_path)

        # 遍历图片文件
        for file_name in os.listdir(new_disaster_folder_path):
            if file_name.endswith('.png'):
                new_file_name = file_name.rsplit('.', 1)[0] + '_target.png'
                file_path = os.path.join(new_disaster_folder_path, file_name)
                new_file_path = os.path.join(new_disaster_folder_path, new_file_name)
                os.rename(file_path, new_file_path)