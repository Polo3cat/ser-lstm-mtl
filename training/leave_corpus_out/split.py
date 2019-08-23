import psycopg2
import argparse
from random import shuffle


'''
    Generates training, validation and test data as ids of the postgres label id.
'''

corpus = ['iemocap', 'aibo', 'emodb', 'enterface', 'ldc']

parser = argparse.ArgumentParser()
parser.add_argument('--corpus', help="Name of corpus to leave out. One of ['iemocap', 'aibo', 'emodb', 'enterface', 'ldc']", required=True)
parser.add_argument('--acted', help='Uses the "acted" instances from the corpus', action='store_true')
parser.add_argument('--validation_ratio', default=0.1, type=float, help='Ratio of training data that is going to be used as validation')
parser.add_argument('--out_dir', help='Output directory, name of output files is creating from corpus option', required=True)
args = parser.parse_args()

conn = psycopg2.connect('postgresql://docker:docker@localhost:5432/features')
cursor = conn.cursor()

cursor.execute('''SELECT id FROM labels WHERE corpus=%s AND acted=%s''', [args.corpus, args.acted])
with open(f"{args.out_dir}/test_ids.txt", 'w') as f:
    for x in cursor:
        f.write(f"{x[0]}\n")

cursor.execute('''SELECT id FROM labels WHERE id NOT IN 
    (SELECT id FROM labels WHERE corpus=%s AND acted=%s AND id IS NOT NULL)''', 
    [args.corpus, args.acted])

rand_ids = [x[0] for x in cursor]
shuffle(rand_ids)
split_idx = int(len(rand_ids)*(1-args.validation_ratio))

with open(f"{args.out_dir}/train_ids.txt", 'w') as f:
    for x in rand_ids[:split_idx]:
        f.write(f"{x}\n")

with open(f"{args.out_dir}/validation_ids.txt", 'w') as f:
    for x in rand_ids[split_idx:]:
        f.write(f"{x}\n")
