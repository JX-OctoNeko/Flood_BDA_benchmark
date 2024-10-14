import os
import re
import pandas as pd
import os.path as osp
import csv

BASE_DIR = osp.dirname(osp.abspath(__file__))

class LogProcessor:
    def __init__(self, base_dir, model_name, set_selection):
        self.base_dir = base_dir
        self.model_name = model_name
        self.set_selection = set_selection
        self.file_folder = osp.join(self.base_dir, self.model_name, self.set_selection)
        self.matches = None
        self.result_dict = {}
        self.data_dict = {}


    def log_match_paragraph(self):
        pattern = r"Evaluate(.*?)Best"
        all_content = ''

        for file_name in os.listdir(self.file_folder):
            if file_name.endswith('.log'):
                file_path = osp.join(self.file_folder,file_name)
                print(file_path)
                with open(file_path, 'r') as file:
                    content = file.read()
                    all_content += content # 内容合并

        self.matches = re.findall(pattern, all_content, re.DOTALL)



    # def store_in_dict(self):
    #     # 储存到字典中
    #     for i, match in enumerate(self.matches):
    #         self.result_dict[i] = match.strip()
    #
    #     return self.result_dict

        # 输出结果
        # print(result_dict)


    # def match_value_lastline(self):
    #     # 用于匹配值中重复出现的结构的正则表达式模式
    #     value_pattern = r"Loss: [\d.]+ \([\d.]+\)\sPrec\. [\d.]+\sRecall [\d.]+\sF1 [\d.]+\sOA [\d.]+"
    #
    #     # 遍历字典中的值，对每个值应用正则表达式
    #     for key, value in self.result_dict.items():
    #         # 使用re.findall找到匹配的所有结构
    #         matches = re.findall(value_pattern, value)
    #         # 如果有匹配的结构，更新字典中的值为最后一组结构
    #         if matches:
    #             self.result_dict[key] = matches[-1]
    #
    #     return self.result_dict
    #
    #     # 输出更新后的字典
    #     # print(result_dict)
    #
    #
    # def value2dict(self):
    #     # 定义用于提取数据的正则表达式模式
    #     data_pattern = r"Loss: ([\d.]+) \(([\d.]+)\)\sPrec\. ([\d.]+)\sRecall ([\d.]+)\sF1 ([\d.]+)\sOA ([\d.]+)"
    #
    #     # 新建一个空字典
    #
    #     # 遍历字典中的值，对每个值进行匹配和提取数据
    #     for key, value in self.result_dict.items():
    #         # 使用re.search找到首次匹配的数据
    #         match = re.search(data_pattern, value)
    #         if match:
    #             # 将提取到的数据转换为字典
    #             self.data_dict[key] = {
    #                 "Loss": match.group(1),
    #                 "Prec": match.group(3),
    #                 "Recall": match.group(4),
    #                 "F1": match.group(5),
    #                 "OA": match.group(6)
    #             }
    #
    #     # 输出转换后的字典
    #     print(self.data_dict)
    #     return self.data_dict

    def process_values2dict(self):
        value_pattern = r"Loss: [\d.]+ \([\d.]+\)\sPrec\. [\d.]+\sRecall [\d.]+\sF1 [\d.]+\sOA [\d.]+"
        data_pattern = r"Loss: ([\d.]+) \(([\d.]+)\)\sPrec\. ([\d.]+)\sRecall ([\d.]+)\sF1 ([\d.]+)\sOA ([\d.]+)"

        # 储存到字典中
        for i, match in enumerate(self.matches):
            self.result_dict[i] = match.strip()

        for key, value in self.result_dict.items():
            matches = re.findall(value_pattern, value)
            if matches:
                match = re.search(data_pattern, matches[-1])
                if match:
                    self.data_dict[key] = {
                        "Loss": match.group(1),
                        "Prec": match.group(3),
                        "Recall": match.group(4),
                        "F1": match.group(5),
                        "OA": match.group(6)
                    }

        # 输出转换后的字典
        print(self.data_dict)

    def write2csv(self):
        # 获取表头，就是字典中第一个epoch键对应的值的键
        headers = ['Epoch'] + list(self.data_dict[next(iter(self.data_dict))].keys())

        # 将数据写入csv文件
        with open(f'{self.model_name}/{self.set_selection}/{self.model_name}_validation.csv', 'w', newline='') as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames=headers)

            # 写入表头
            csv_writer.writeheader()

            # 遍历字典，写入csv
            for epoch, values in self.data_dict.items():
                csv_writer.writerow({'Epoch': epoch, **values})

    def operator(self):
        self.log_match_paragraph()
        self.process_values2dict()
        self.write2csv()


if __name__ == '__main__':

    bit_base_val = LogProcessor(BASE_DIR, 'bit_base', 'val')
    bit_base_val.operator()

    # p2v60 = LogProcessor(BASE_DIR, 'p2v', 'val')
    # p2v60.operator()

    # snunet60 = LogProcessor(BASE_DIR, 'snunet', 'val')
    # snunet60.operator()

    # bit60 = LogProcessor(BASE_DIR, 'bit60', 'val')
    # bit60.operator()
    #
    # bit90 = LogProcessor(BASE_DIR, 'bit90', 'val')
    # bit90.operator()