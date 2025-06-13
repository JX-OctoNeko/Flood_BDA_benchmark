import os
from PIL import Image

def split_images(input_folder, output_folder, num_slices):
    # 遍历输入文件夹中的所有图片
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            # 构建完整的输入图片路径
            input_path = os.path.join(input_folder, filename)

            # 打开原始图片
            original_image = Image.open(input_path)

            # 获取原始图片的宽度和高度
            original_width, original_height = original_image.size

            # 计算每个切割区域的宽度
            slice_width = original_width // num_slices

            # 创建子文件夹，按照图片名字分类
            output_subfolder = os.path.join(output_folder, os.path.splitext(filename)[0])
            os.makedirs(output_subfolder, exist_ok=True)

            # 逐个切割并保存小图片到子文件夹中
            for i in range(num_slices):
                # 计算切割的区域
                left = i * slice_width
                upper = 0
                right = (i + 1) * slice_width
                lower = original_height

                # 切割图片
                slice_image = original_image.crop((left, upper, right, lower))

                # 构建完整的输出图片路径
                output_path = os.path.join(output_subfolder, f"slice_{i + 1}.png")

                # 保存切割后的小图片
                slice_image.save(output_path)

# 调用函数进行切割，例如将图片文件夹中的所有图片分成5份
split_images("./for_caadria_pic", "./slice_pic", 8)