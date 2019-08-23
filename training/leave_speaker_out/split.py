import psycopg2
import argparse
from random import shuffle


'''
    Generates training, validation and test data as ids of the postgres label id.
'''

parser = argparse.ArgumentParser()
parser.add_argument('--corpus', help="Name of corpus to leave out. One of ['iemocap', 'aibo', 'emodb', 'enterface', 'ldc']", required=True)
parser.add_argument('--acted', help='Uses the "acted" instances from the corpus', action='store_true')
parser.add_argument('--validation_ratio', default=0.1, type=float, help='Ratio of training data that is going to be used as validation')
parser.add_argument('--out_dir', help='Output directory, name of output files is creating from corpus option', required=True)
args = parser.parse_args()

conn = psycopg2.connect('postgresql://docker:docker@localhost:5432/features')
cursor = conn.cursor()

cursor.execute('''WITH N AS (select count(*) from labels where corpus=%s and acted=%s)
    SELECT id, speaker_number FROM labels 
    WHERE corpus=%s AND acted=%s AND speaker_number=(SElECT speaker_number FROM labels WHERE corpus=%s and acted=%s OFFSET floor(random()*(select * from N)) LIMIT 1)''', 
    [args.corpus, args.acted]*3)

result = cursor.fetchall()
with open(f"{args.out_dir}/test_ids.txt", 'w') as f:
    for x,_ in result:
        f.write(f"{x}\n")

cursor.execute('''SELECT id FROM labels WHERE corpus=%s AND acted=%s AND NOT speaker_number=%s''', 
    [args.corpus, args.acted, result[0][1]])

rand_ids = [x[0] for x in cursor]
shuffle(rand_ids)
split_idx = int(len(rand_ids)*(1-args.validation_ratio))

with open(f"{args.out_dir}/train_ids.txt", 'w') as f:
    for x in rand_ids[:split_idx]:
        f.write(f"{x}\n")

with open(f"{args.out_dir}/validation_ids.txt", 'w') as f:
    for x in rand_ids[split_idx:]:
        f.write(f"{x}\n")
