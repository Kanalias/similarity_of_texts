import pandas
import seaborn
from matplotlib import pyplot as plt


class Plot:

    def show_heatmap(self, dataframe: pandas.DataFrame, file_name: str, title: str, cmap: str = None, figsize: tuple = (16, 6)):
        plt.figure(figsize=figsize)
        heatmap = seaborn.heatmap(dataframe, vmin=0, vmax=1, annot=True, linewidths=2, linecolor='black', square=True,
                                  cmap=cmap)
        heatmap.set_title(title, fontdict={'fontsize': 18}, pad=16)

        plt.savefig(file_name, dpi=300, bbox_inches='tight')