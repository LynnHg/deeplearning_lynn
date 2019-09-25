import matplotlib.pyplot as plt
import numpy as np
from dataset import Dataset

dst = Dataset()

imgs, labels = next(iter(dst.data_loader_test))

size = 4
batch_imgs = [0] * size
mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
for i in range(size):
    img = imgs[i].numpy().transpose(1, 2, 0)
    img = img * std + mean
    batch_imgs[i] = img

category_names = ['acc', 'error']
results = [
    {
    'a': [91, 8],
    'b': [98.22, 1.22],
    'c': [93, 7],
    'd': [95.22, 3.22],
    'e': [98.22, 1.22]
    },
    {
    'a': [92, 8],
    'b': [98.22, 1.22],
    'c': [93, 7],
    'd': [95.22, 3.22],
    'f': [98.22, 10.22]
    },
    {
        'a': [93, 8],
        'b': [98.22, 1.22],
        'c': [93, 7],
        'd': [95.22, 3.22],
        'e': [120.22, 1.22]
    },
    {
        'a': [94, 8],
        'b': [98.22, 1.22],
        'c': [93, 7],
        'd': [95.22, 3.22],
        'e': [102.22, 1.22]
    }
]



def show_top_data(top1_data, top5_data, results, category_names):
    # top1 figure
    fig1 = plt.figure()
    fig1.subplots_adjust(0.1, 0.11, 0.79, 0.88, 0.03, 1)
    # top5 figure
    fig5 = plt.figure()
    fig5.subplots_adjust(0.1, 0.11, 0.79, 0.88, 0.03, 1)
    j = 1
    while j < size * 2:
        for i in range(size):
            labels = list(results[i].keys())
            data = np.array(list(results[i].values()))
            data_cum = data.cumsum(axis=1)
            category_colors = plt.get_cmap('RdYlGn')(
                np.linspace(0.85, 0.15, data.shape[1]))

            # top5 acc
            ax2 = fig5.add_subplot(size, 2, j)
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['bottom'].set_visible(False)
            ax2.spines['left'].set_visible(False)
            ax2.imshow(batch_imgs[i])
            ax = fig5.add_subplot(size, 2, j + 1)
            ax.invert_yaxis()
            ax.xaxis.set_visible(False)
            ax.set_xlim(0, np.sum(data, axis=1).max())
            j += 2
            for i, (colname, color) in enumerate(zip(category_names, category_colors)):
                widths = data[:, i]
                starts = data_cum[:, i] - widths
                ax.barh(labels, widths, left=starts, height=0.8,
                        label=colname, color=color)
                xcenters = starts + widths / 2

                r, g, b, _ = color
                text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
                for y, (x, c) in enumerate(zip(xcenters, widths)):
                    ax.text(x, y, str(int(c))+'%', ha='center', va='center',
                            color=text_color)
            ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
                      loc='lower left', fontsize='small')

    plt.subplots_adjust(0.27, 0.11, 0.68, 0.88, 0, 1)
    plt.show()
    # return fig, ax


show_top_data(1, 2, results, category_names)

