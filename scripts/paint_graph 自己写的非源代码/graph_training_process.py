import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import os.path as osp
from matplotlib.font_manager import FontProperties

BASE_DIR = osp.dirname(osp.abspath(__file__))
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = 'Times New Roman'
font_title = {
    'fontsize': 16,
    'fontweight': 'bold'
}
font_legend = {
    'fontsize': 14,
}
font_label = {
    'fontsize': 14,
}

class GraphVal():
    def __init__(self, base_dir, model_names, set_selection, metrics, iteration):
        self.base_dir = base_dir
        self.model_names = model_names
        self.set_selection = set_selection
        self.data = None
        self.metrics = metrics
        self.iteration = iteration


    def read_csv_file(self, model_name):
        file_folder = osp.join(self.base_dir, model_name, self.set_selection)
        self.data = pd.read_csv(f"{file_folder}/{model_name}_validation.csv")


    def plot_metrics(self, model_name):
        model_name = model_name.upper()
        plt.plot(self.data[f'{self.iteration}'].values, self.data[f'{self.metrics}'].values, label=model_name)

    def operator(self):
        for model_name in self.model_names:
            self.read_csv_file(model_name)
            self.plot_metrics(model_name)

    def show_total_graph(self):
        plt.title(f'{self.iteration} vs {self.metrics}-score', fontdict=font_title)
        plt.xlabel(f'{self.iteration}', fontdict=font_label)
        plt.ylabel(f'{self.metrics}-score', fontdict=font_label)
        plt.legend()
        plt.show()

        # figure = plt.figure()
        # ax = figure.add_subplot(111)
        # ax.set_title(f'{self.iteration} vs {self.metrics}ision', fontproperties=font_title)
        # ax.set_xlabel(f'{self.iteration}', fontproperties=font_label)
        # ax.set_ylabel(f'{self.metrics}ision', fontproperties=font_label)
        # ax.legend(loc='best')
        # plt.show()





if __name__ == '__main__':
    model_names = ['bit_modified', 'bit_base', 'snunet', 'p2v']

    graph = GraphVal(BASE_DIR, model_names, 'val', 'Recall', 'Epoch')
    graph.operator()
    graph.show_total_graph()

    test_model_names = ['bit_modified', 'bit_base', 'snunet', 'p2v']
    test_graph = GraphVal(BASE_DIR, test_model_names, 'test', 'Recall', 'Batch')
    test_graph.operator()
    test_graph.show_total_graph()


