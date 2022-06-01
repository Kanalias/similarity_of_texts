import pandas
import seaborn
from matplotlib import pyplot as plt
import random

class Plot:

    def show_heatmap(self, dataframe: pandas.DataFrame, file_name: str, title: str, cmap: str = None, figsize: tuple = (16, 6)):
        plt.figure(figsize=figsize)
        heatmap = seaborn.heatmap(dataframe, vmin=0, vmax=1, annot=True, linewidths=2, linecolor='black', square=True,
                                  cmap=cmap)
        heatmap.set_title(title, fontdict={'fontsize': 18}, pad=16)

        plt.savefig(file_name, dpi=300, bbox_inches='tight')

    def show_word_embeding(self, x_vals, y_vals, labels, *, file_name: str = "wv.png", selected_indices: int = 50):

        random.seed(0)

        plt.figure(figsize=(12, 12))
        plt.scatter(x_vals, y_vals)

        indices = list(range(len(labels)))
        selected_indices = random.sample(indices, selected_indices)
        for i in selected_indices:
            plt.annotate(labels[i], (x_vals[i], y_vals[i]))

        plt.savefig(file_name, dpi=300, bbox_inches='tight')