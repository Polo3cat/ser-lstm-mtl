import argparse
import pickle
from collections import defaultdict
from pprint import pprint
from itertools import combinations

import numpy as np
from scipy.stats import wilcoxon


if __name__ == "__main__":
  np.set_printoptions(precision=2)
  parser = argparse.ArgumentParser()
  parser.add_argument('inputs', help='List of pickle files with an ndarray representing a confusion matrix.')
  args = parser.parse_args()
  table = defaultdict(lambda: defaultdict(dict))
  with open(args.inputs) as g:
      for l in g:
        with open(l.rstrip(), 'rb') as f:
          _cm = pickle.load(f)
          name = l.split('/')[-1].split('.')[0].split('_')[0]
          protocol = l.split('/')[-3]
          corpus = l.split('/')[-2]
          correct = np.trace(_cm)
          total = np.sum(_cm)
          table[protocol][corpus][name] = correct / total
  pprint(table)
  
  print('Wilcoxon signed-rank test')
  for test_type, corpora in table.items():
    accuracies = defaultdict(list)
    for _, corpus in corpora.items():
      for architecture, accuracy in corpus.items():
        accuracies[architecture].append(accuracy)
    print(test_type)
    for (namex, x),(namey, y) in combinations(accuracies.items(), 2):
      print(f"{namex} - {namey}")
      _, p_value = wilcoxon(x,y, zero_method="pratt", correction=False)
      print("{:.3f}".format(p_value))
      print()
  for ttype, datasets in table.items():
    print(ttype)
    average = defaultdict(int)
    for corpus, res in datasets.items():
      #print(res.keys())
      template_row = R'\textbf{{{}}}' + ' & {:.3f}'*len(res.values()) + R'\\'
      print(template_row.format(corpus, *res.values()))
      for k,v in res.items():
        average[k] += v
    print(R'\textbf{{Average}} ' + ('& {:.3f}'*len(average.values())).format(*[x/6 for x in average.values()]) + R'\\')