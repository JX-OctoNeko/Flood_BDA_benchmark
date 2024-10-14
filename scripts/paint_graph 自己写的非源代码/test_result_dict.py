import os
import re
import os.path as osp
import csv


BASE_DIR = osp.dirname(osp.abspath(__file__))

class TestProcessor:

    def __init__(self, base_dir, model_name, mode):
        self.base_dir = base_dir
        self.model_name = model_name
        self.mode = mode
        self.content = None
        self.data_dict = {}


    def read_testlog_file(self):
        file_folder = osp.join(self.base_dir, self.model_name, self.mode)
        all_content = ''
        for file in os.listdir(file_folder):
            if file.endswith('.log'):
                file_path = osp.join(file_folder, file)
                with open(file_path, 'r') as file:
                    content = file.read()
                    all_content += content

        self.content = all_content


    def value2dict(self):
        data_dict = {}
        # 提取所有行
        value_pattern = r"Loss: [\d.]+ \([\d.]+\)\sPrec\. [\d.]+\sRecall [\d.]+\sF1 [\d.]+\sOA [\d.]+"
        # 提取行中数值
        data_pattern = r"Loss: ([\d.]+) \(([\d.]+)\)\sPrec\. ([\d.]+)\sRecall ([\d.]+)\sF1 ([\d.]+)\sOA ([\d.]+)"

        matches = re.findall(value_pattern, self.content, re.DOTALL)

        # 创建一个键值对储存文本作为值
        for i, match in enumerate(matches):
            values = match.strip()
            value_match = re.search(data_pattern, values)

            if value_match:
                # 转换为字典
                data_dict[i] = {
                    "Loss": value_match.group(2),
                    "Prec": value_match.group(3),
                    "Recall": value_match.group(4),
                    "F1": value_match.group(5),
                    "OA": value_match.group(6)
                }


        print(data_dict)
        self.data_dict = data_dict


    def write2csv(self):
        headers = ['Batch'] + list(self.data_dict[next(iter(self.data_dict))].keys())

        # 将数据写入csv文件
        with open(f'{self.model_name}/{self.mode}/{self.model_name}_validation.csv', 'w', newline='') as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames=headers)

            # 写入表头
            csv_writer.writeheader()

            # 遍历字典，写入csv
            for batch, values in self.data_dict.items():
                csv_writer.writerow({'Batch': batch, **values})

    def operator(self):
        self.read_testlog_file()
        self.value2dict()
        self.write2csv()


if __name__ == "__main__":

    # snunet = TestProcessor(BASE_DIR,'snunet','test')
    # snunet.operator()
    #
    # p2v = TestProcessor(BASE_DIR,'p2v','test')
    bitbase = TestProcessor(BASE_DIR, 'bit_base', 'test')
    # bit_modified = TestProcessor(BASE_DIR, 'bit_modified', 'test')
    # bit90 = TestProcessor(BASE_DIR, 'bit90', 'test')

    # p2v.operator()
    bitbase.operator()
    # bit_modified.operator()
    # bit90.operator()
