import argparse
import pickle
from collections import defaultdict
from pprint import pprint

import numpy as np


if __name__ == "__main__":
  np.set_printoptions(precision=2)
  parser = argparse.ArgumentParser()
  parser.add_argument('inputs', help='List of pickle files with an ndarray representing a confusion matrix.')
  args = parser.parse_args()
  table = defaultdict(lambda: defaultdict(dict))
  average = defaultdict(lambda: defaultdict(lambda: np.zeros((4,4), dtype=float)))
  with open(args.inputs) as g:
      for l in g:
        with open(l.rstrip(), 'rb') as f:
          _cm = pickle.load(f)
          name = l.split('/')[-1].split('.')[0]
          protocol = l.split('/')[-3]
          corpus = l.split('/')[-2]
          correct = np.trace(_cm)
          total = np.sum(_cm)
          table[protocol][corpus][name] = correct / total
          average[protocol][name] += _cm
  pprint(table)
  for k, pr in average.items():
    for k2, na in pr.items():
      avg = na / 4
      correct = np.trace(avg)
      total = np.sum(avg)
      print(k, k2, correct / total)
  