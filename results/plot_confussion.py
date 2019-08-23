import argparse
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(cm,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    classes = ['anger', 'hapiness', 'sadness', 'neutral']
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    if normalize:
        b = cm.sum(axis=1)
        cm = np.nan_to_num(cm / b[:, np.newaxis])
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig

if __name__ == "__main__":
  np.set_printoptions(precision=2)
  parser = argparse.ArgumentParser()
  parser.add_argument('inputs', help='List of pickle files with an ndarray representing a confusion matrix.')
  args = parser.parse_args()
  average = defaultdict(lambda: defaultdict(lambda: np.zeros((4,4), dtype=float)))
  with open(args.inputs) as g:
      for l in g:
        with open(l.rstrip(), 'rb') as f:
          _cm = pickle.load(f)
          print(_cm)
          name = l.split('/')[-1].split('.')[0]
          protocol = l.split('/')[-3]
          corpus = l.split('/')[-2]
          fig = plot_confusion_matrix(_cm, normalize=True, title=f"{protocol}_{corpus}_{name}"[:-22])
          fig.savefig(f"figures/{protocol}_{corpus}_{name}.png", bbox_inches='tight', dpi=400)
          plt.close(fig)
          average[protocol][name] += _cm

  for k, pr in average.items():
    for k2, na in pr.items():
      avg = na / 4
      fig = plot_confusion_matrix(avg, normalize=True, title=f"Average {k} {k2}")
      fig.savefig(f"figures/average_{k}_{k2}.png", bbox_inches='tight', dpi=400)
      plt.close(fig)
